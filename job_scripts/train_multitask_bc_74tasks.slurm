#!/bin/bash
#SBATCH --job-name=trainbc
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out



set -x
set -e

module purge
pwd; hostname; date

code_dir=$WORK/Code/polarnet
export PYTHONPATH="$PYTHONPATH:$code_dir"

. $WORK/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH

conda activate polarnet
export PYTHONPATH=$PYTHONPATH:$(pwd)

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

echo $SINGULARITY_ALLOWED_DIR
export SINGULARITY_ALLOWED_DIR=/gpfsssd/singularity/images/uqn73qm
module load singularity

seed=0
taskvars=$code_dir/polarnet/assets/74_tasks_var.csv
checkpoint=pretrained_models/pointnext-s-c64-enc-dec-sameshape.pt
output_dir=exprs/74tasks-multi-model/seed${seed}/
config_file=$code_dir/polarnet/config/74tasks.yaml
data_dir=data/74tasks/keysteps_pcd/seed${seed}/
instr_embed_file=data/taskvar_instrs/clip/

pushd $code_dir/polarnet
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
    $SINGULARITY_ALLOWED_DIR/polarnet.sif \
    xvfb-run -a python train_models.py \
    --exp-config ${config_file} --restart_epoch 0 \
	output_dir ${output_dir} \
    DATASET.dataset_class pre_pcd_keystep_stepwise \
    DATASET.taskvars $taskvars \
    DATASET.in_memory False n_workers 8 \
	DATASET.data_dir  $data_dir \
    DATASET.instr_embed_file $instr_embed_file \
    DATASET.exclude_overlength_episodes 20 \
	num_train_steps 1000000 save_steps 10000 \
	MODEL.dropout 0.0 DATASET.color_drop 0.0 \
	checkpoint_strict_load False \
	MODEL.num_trans_layers 2 train_batch_size 8 \
    checkpoint $checkpoint \
    MODEL.pcd_encoder_cfg.width 64 \
    MODEL.learnable_step_embedding False \
    MODEL.pcd_encoder_cfg.in_channels 10 \
    MODEL.use_prev_action True 
popd
