from typing import List

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from scipy.spatial.transform import Rotation as R

from openpoints.models.backbone.pointnext import (
    PointNextEncoder, FeaturePropogation
)

from polarnet.models.network_utils import (
    dense_layer, normalise_quat, ActionLoss, PositionalEncoding
)
from polarnet.models.base import BaseModel


class PointNextDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]     # feature propogation

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i]
            )
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f, txt_tokens=None, txt_padding_masks=None, return_all_layers=False):
        if return_all_layers:
            out_per_layer = []

        for i in range(-1, -len(self.decoder) - 1, -1):
            x = self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])
            f[i - 1] = self.decoder[i][1:]([p[i], x])[1]
            if return_all_layers:
                out_per_layer.append(f[i - 1])

        if return_all_layers:
            return out_per_layer

        out = f[-len(self.decoder) - 1]
        return out


class ActionHead(nn.Module):
    def __init__(
            self, dec_channels, heatmap_temp=1, dropout=0, use_max_action=False,
            use_discrete_rot:bool=False, rot_resolution:int=5,
        ) -> None:
        super().__init__()
        self.use_discrete_rot = use_discrete_rot
        self.rot_resolution = rot_resolution

        if self.use_discrete_rot:
            self.rot_decoder = nn.Sequential(
                *dense_layer(dec_channels[0], dec_channels[0] // 2),
                nn.Dropout(dropout),
                *dense_layer(dec_channels[0] // 2, (360 // rot_resolution) * 3 + 1, apply_activation=False),
            )
        else:
            self.quat_decoder = nn.Sequential(
                *dense_layer(dec_channels[0], dec_channels[0] // 2),
                nn.Dropout(dropout),
                *dense_layer(dec_channels[0] // 2, 4 + 1, apply_activation=False),
            )

        self.maps_to_coord = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dec_channels[-1], 1 + 3, 1)
        )
        self.heatmap_temp = heatmap_temp
        self.use_max_action = use_max_action

    def forward(self, dec_fts, pcds, pc_centers, pc_radii):
        '''
        - dec_fts: [(batch, dec_channels[0], npoints), (batch, dec_channels[-1], npoints)]
        - pcds: (batch, 3, npoints)
        '''
        # predict the translation of the gripper
        xt_fts = self.maps_to_coord(dec_fts[-1])
        xt_heatmap = torch.softmax(xt_fts[:, :1] / self.heatmap_temp, dim=-1)
        xt_offset = xt_fts[:, 1:]
        if self.use_max_action:
            xt = pcds + xt_offset   # (b, 3, npoints)
            xt = xt.gather(
                2, einops.repeat(torch.max(xt_heatmap, dim=2)[1], 'b 1 -> b 3').unsqueeze(2)
            ).squeeze(2)
        else:
            xt = einops.reduce((pcds + xt_offset) * xt_heatmap, 'b c n -> b c', 'sum')
        xt = xt * pc_radii + pc_centers

        # predict the (rotation, openness) of the gripper
        xg_fts, _ = torch.max(dec_fts[0], -1)

        if self.use_discrete_rot:
            xg = self.rot_decoder(xg_fts)
            xr = xg[..., :-1]
        else:
            xg = self.quat_decoder(xg_fts)
            xr = normalise_quat(xg[..., :4])
        
        xo = xg[..., -1:]

        actions = torch.cat([xt, xr, xo], dim=-1)

        return {
            'actions': actions, 
            'xt_offset': xt_offset * pc_radii.unsqueeze(2), 
            'xt_heatmap': xt_heatmap.squeeze(1),
            'xt_heatmap_logits': xt_fts[:, 0] / self.heatmap_temp,
        }


class ActionEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.open_embedding = nn.Embedding(2, hidden_size)
        self.pos_embedding = nn.Linear(3, hidden_size)
        self.rot_embedding = nn.Linear(6, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, actions):
        '''
        actions: (batch_size, 8)
        '''
        pos_embeds = self.pos_embedding(actions[..., :3])
        open_embeds = self.open_embedding(actions[..., -1].long())

        rot_euler_angles = R.from_quat(actions[..., 3:7].data.cpu()).as_euler('xyz')
        rot_euler_angles = torch.from_numpy(rot_euler_angles).float().to(actions.device)
        rot_inputs = torch.cat(
            [torch.sin(rot_euler_angles), torch.cos(rot_euler_angles)], -1
        )
        rot_embeds = self.rot_embedding(rot_inputs)

        act_embeds = self.layer_norm(
            pos_embeds + rot_embeds + open_embeds
        )
        return act_embeds


class PointCloudUNet(BaseModel):
    def __init__(
        self, pcd_encoder_cfg, pcd_decoder_cfg,
        num_tasks: int = None, max_steps: int = 20,
        use_instr_embed: str = 'none', instr_embed_size: int = None,
        txt_attn_type: str = 'none', num_trans_layers: int = 1,
        trans_hidden_size: int = 512,
        dropout=0.2, heatmap_temp=1, use_prev_action=False, 
        cat_global_in_head=False, **kwargs
    ):
        super().__init__()

        self.pcd_encoder_cfg = pcd_encoder_cfg
        self.pcd_decoder_cfg = pcd_decoder_cfg
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.use_instr_embed = use_instr_embed
        self.instr_embed_size = instr_embed_size
        self.txt_attn_type = txt_attn_type
        self.num_trans_layers = num_trans_layers
        self.use_prev_action = use_prev_action
        self.cat_global_in_head = cat_global_in_head
        self.heatmap_temp = heatmap_temp
        self.use_discrete_rot = kwargs.get('use_discrete_rot', False)
        self.rot_resolution = kwargs.get('rot_resolution', 5)
        self.kwargs = kwargs

        self.pcd_encoder = PointNextEncoder(**pcd_encoder_cfg)
        enc_channel_list = self.pcd_encoder.channel_list
        self.hidden_size = trans_hidden_size

        self.pcd_decoder = PointNextDecoder(
            enc_channel_list[:-1] + [enc_channel_list[-1] + self.hidden_size], pcd_decoder_cfg.layers,
        )

        if self.kwargs.get('learnable_step_embedding', True):
            self.step_embedding = nn.Embedding(self.max_steps, self.hidden_size)
        else:
            self.step_embedding = PositionalEncoding(self.hidden_size, max_len=self.max_steps)

        if self.use_prev_action:
            self.prev_action_embedding = ActionEmbedding(self.hidden_size)

        if self.use_instr_embed == 'none':
            assert self.num_tasks is not None
            self.task_embedding = nn.Embedding(self.num_tasks, self.hidden_size)
        else:
            assert self.instr_embed_size is not None
            self.task_embedding = nn.Linear(self.instr_embed_size, self.hidden_size)

        self.point_pos_embedding = nn.Linear(3, self.hidden_size)

        if self.txt_attn_type == 'cross':
            if enc_channel_list[-1] != self.hidden_size:
                self.pcd_to_trans_fc = nn.Conv1d(
                    in_channels=enc_channel_list[-1], 
                    out_channels=self.hidden_size, 
                    kernel_size=1, stride=1
                )
            else:
                self.pcd_to_trans_fc = None
            trans_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size*4,
                dropout=0.1, activation='gelu',
                layer_norm_eps=1e-12, norm_first=False,
                batch_first=True,
            )
            self.cross_attention = nn.TransformerDecoder(
                trans_layer, num_layers=self.num_trans_layers
            )

        dec_ft_size = enc_channel_list[0]
        if self.cat_global_in_head:
            dec_ft_size += self.hidden_size
        self.head = ActionHead(
            [enc_channel_list[-1] + self.hidden_size, dec_ft_size], 
            heatmap_temp=heatmap_temp, dropout=dropout,
            use_max_action=kwargs.get('use_max_action', False),
            use_discrete_rot=self.use_discrete_rot,
            rot_resolution=self.rot_resolution,
        )

        self.loss_fn = ActionLoss(
            use_discrete_rot=self.use_discrete_rot, 
            rot_resolution=self.rot_resolution
        )

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch(batch)

        # encode point cloud
        pcd_fts = batch['fts']  # (batch, dim, npoints)
        pcd_poses = pcd_fts[:, :3]

        pos_list, ft_list = self.pcd_encoder(
            pcd_poses.permute(0, 2, 1).contiguous(), pcd_fts
        )
        ctx_embeds = ft_list[-1]
        if self.pcd_to_trans_fc is not None:
            ctx_embeds = self.pcd_to_trans_fc(ctx_embeds)

        step_ids = batch['step_ids']
        step_embeds = self.step_embedding(step_ids)
        ctx_embeds = ctx_embeds + step_embeds.unsqueeze(2)
        if self.use_prev_action:
            ctx_embeds = ctx_embeds + self.prev_action_embedding(batch['prev_actions']).unsqueeze(2)
        ctx_embeds = ctx_embeds + self.point_pos_embedding(pos_list[-1]).permute(0, 2, 1)

        # conditioned on the task
        taskvar_ids = batch['taskvar_ids']
        instr_embeds = batch.get('instr_embeds', None)
        txt_masks = batch.get('txt_masks', None)

        if self.use_instr_embed == 'none':
            task_embeds = self.task_embedding(taskvar_ids).unsqueeze(1)  # (batch, 1, dim)
        else:
            task_embeds = self.task_embedding(instr_embeds) # (batch, 1/len, dim)
        
        if self.txt_attn_type == 'none':
            assert task_embeds.size(1) == 1
            ctx_embeds = ctx_embeds + task_embeds.permute(0, 2, 1)
        elif self.txt_attn_type == 'cross':
            assert txt_masks is not None
            ctx_embeds = self.cross_attention(
                ctx_embeds.permute(0, 2, 1), task_embeds,
                memory_key_padding_mask=txt_masks.logical_not(),
            )
            ctx_embeds = ctx_embeds.permute(0, 2, 1)
        else:
            raise NotImplementedError(f'unsupported txt_attn_type {self.txt_attn_type}')

        ft_list[-1] = torch.cat([ft_list[-1], ctx_embeds], dim=1)

        # decoding features
        dec_fts = self.pcd_decoder(pos_list, ft_list)

        if self.cat_global_in_head:
            global_ctx_embeds, _ = torch.max(ctx_embeds, 2)
            global_ctx_embeds = einops.repeat(global_ctx_embeds, 'b c -> b c n', n=dec_fts.size(2))
            dec_fts = torch.cat([dec_fts, global_ctx_embeds], dim=1)
        outs = self.head(
            (ft_list[-1], dec_fts), pcd_poses,
            batch['pc_centers'], batch['pc_radii']
        )
        actions = outs['actions']
        
        if compute_loss:
            heatmap_loss = self.kwargs.get('heatmap_loss', False)
            heatmap_loss_weight = self.kwargs.get('heatmap_loss_weight', 1)
            distance_weight = self.kwargs.get('heatmap_distance_weight', 1)
            if heatmap_loss:
                pcd_xyzs = pcd_poses.permute(0, 2, 1) * batch['pc_radii'].unsqueeze(1) + batch['pc_centers'].unsqueeze(1) # (b, npoints, 3)
            else:
                pcd_xyzs = None
            losses = self.loss_fn.compute_loss(
                actions, batch['actions'], heatmap_loss=heatmap_loss, 
                pred_heatmap_logits=outs['xt_heatmap_logits'], 
                pred_offset=outs['xt_offset'],
                pcd_xyzs=pcd_xyzs, distance_weight=distance_weight,
                heatmap_loss_weight=heatmap_loss_weight,
                use_heatmap_max=self.kwargs.get('use_heatmap_max', False), 
                use_pos_loss=self.kwargs.get('use_pos_loss', True)
            )
            
            return losses, actions

        return actions
