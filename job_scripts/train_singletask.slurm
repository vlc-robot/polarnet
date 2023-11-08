#!/bin/bash
#SBATCH --job-name=trainbc
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

set -x
set -e

module purge
pwd; hostname; date

code_dir=$WORK/Code/polarnet/
export PYTHONPATH="$PYTHONPATH:$code_dir"

. $WORK/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH

conda activate polarnet

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

module load singularity

seed=0
taskvar=pick_and_lift+0
checkpoint=data/pretrained_models/pointnext-s-c64-enc-dec-sameshape.pt
output_dir=data/exprs/${taskvar}_model/seed${seed}
config_file=$code_dir/polarnet/config/single_task.yaml
instr_embed_file=data/10tasks_data/taskvar_instrs/clip/
data_dir=data/10tasks_data/train/keysteps_pcd/seed${seed}

pushd $code_dir/polarnet
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
    $SINGULARITY_ALLOWED_DIR/polarnet.sif \
    xvfb-run -a python train_models.py \
    --exp-config $config_file \
	output_dir  $output_dir \
    DATASET.taskvars ${taskvar} \
	DATASET.data_dir  ${data_dir} \
    DATASET.instr_embed_file  $instr_embed_file \
	num_train_steps 100000 save_steps 2000 \
	checkpoint $checkpoint 
popd
