#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
##SBATCH --qos=qos_gpu-dev
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

# This script generates fresh samples
# OOM errors with CPU

# go into the submission directory
set -x
set -e

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=/opt/YARR/

code_dir=$WORK/Code/polarnet/

log_dir=$WORK/logs
data_dir=data/10tasks_data/

mkdir -p $log_dir

module load singularity

image=polarnet.sif

export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:$code_dir"

pushd $code_dir/polarnet
# generate preprocessed keysteps
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
	$SINGULARITY_ALLOWED_DIR/${image} \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
  /usr/bin/python3.9 preprocess/generate_pcd_dataset_keysteps.py \
    --seed ${seed} \
    --dataset_dir $data_dir/train/ \
    --outname $outname
popd
