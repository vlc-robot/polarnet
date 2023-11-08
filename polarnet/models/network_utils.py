from typing import Optional, Tuple, Literal, Union, List, Dict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class ActionLoss(object):
    def __init__(self, use_discrete_rot: bool = False, rot_resolution: int = 5):
        self.use_discrete_rot = use_discrete_rot
        if self.use_discrete_rot:
            self.rot_resolution = rot_resolution
            self.rot_classes = 360 // rot_resolution

    def decompose_actions(self, actions, onehot_rot=False):
        pos = actions[..., :3]
        if not self.use_discrete_rot:
            rot = actions[..., 3:7]
            open = actions[..., 7]
        else:
            if onehot_rot:
                rot = actions[..., 3: 6].long()
            else:
                rot = [
                    actions[..., 3: 3 + self.rot_classes],
                    actions[..., 3 + self.rot_classes: 3 + 2*self.rot_classes],
                    actions[..., 3 + 2*self.rot_classes: 3 + 3*self.rot_classes],
                ]
            open = actions[..., -1]
        return pos, rot, open

    def compute_loss(
            self, preds, targets, masks=None,
            heatmap_loss=False, distance_weight=1, heatmap_loss_weight=1,
            pred_heatmap_logits=None, pred_offset=None, pcd_xyzs=None,
            use_heatmap_max=False, use_pos_loss=True
        ) -> Dict[str, torch.Tensor]:
        pred_pos, pred_rot, pred_open = self.decompose_actions(preds)
        tgt_pos, tgt_rot, tgt_open = self.decompose_actions(targets, onehot_rot=True)        

        losses = {}
        losses['pos'] = F.mse_loss(pred_pos, tgt_pos)

        if self.use_discrete_rot:
            losses['rot'] = (F.cross_entropy(pred_rot[0], tgt_rot[:, 0]) + \
                            F.cross_entropy(pred_rot[1], tgt_rot[:, 1]) + \
                            F.cross_entropy(pred_rot[2], tgt_rot[:, 2])) / 3
        else:
            # Automatically matching the closest quaternions (symmetrical solution).
            tgt_rot_ = -tgt_rot.clone()
            rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
            select_mask = (rot_loss < rot_loss_).float()
            losses['rot'] = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()

        losses['open'] = F.binary_cross_entropy_with_logits(pred_open, tgt_open)

        if use_pos_loss:
            losses['total'] = losses['pos'] + losses['rot'] + losses['open']
        else:
            losses['total'] = losses['rot'] + losses['open']

        if heatmap_loss:
            # (batch, npoints, 3)
            tgt_offset = targets[:, :3].unsqueeze(1) - pcd_xyzs 
            dists = torch.norm(tgt_offset, dim=-1)
            if use_heatmap_max:
                tgt_heatmap_index = torch.min(dists, 1)[1]  # (b, )
                
                losses['xt_heatmap'] = F.cross_entropy(
                    pred_heatmap_logits, tgt_heatmap_index
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = F.mse_loss(
                    pred_offset.gather(
                        2, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(2)
                    ),
                    tgt_offset.gather(
                        1, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(1)
                    )
                )
                losses['total'] += losses['xt_offset']

            else:
                inv_dists = 1 / (1e-12 + dists)**distance_weight

                tgt_heatmap = torch.softmax(inv_dists, dim=1)
                tgt_log_heatmap = torch.log_softmax(inv_dists, dim=1)
                losses['tgt_heatmap_max'] = torch.mean(tgt_heatmap.max(1)[0])

                losses['xt_heatmap'] = F.kl_div(
                    torch.log_softmax(pred_heatmap_logits, dim=-1), tgt_log_heatmap,
                    reduction='batchmean', log_target=True
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = torch.sum(F.mse_loss(
                    pred_offset.permute(0, 2, 1), tgt_offset, 
                    reduction='none'
                ) * tgt_heatmap.unsqueeze(2)) / tgt_offset.size(0) / 3

                losses['total'] += losses['xt_offset']

        return losses
    
    
class PositionalEncoding(nn.Module):
    '''
        Transformer-style positional encoding with wavelets
    '''

    def __init__(self, dim_embed, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim_embed, 2, dtype=torch.float) *
                -(math.log(10000.0) / dim_embed)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        self.pe = pe # size=(max_len, dim_embed)
        self.dim_embed = dim_embed

    def forward(self, step_ids):
        if step_ids.device != self.pe.device:
            self.pe = self.pe.to(step_ids.device)
        return self.pe[step_ids]
