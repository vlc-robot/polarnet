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
#SBATCH --array=0-9

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
num_episodes=${num_episodes:-100}
task_offset=${task_offset:-1}
seed_default=${seed_default:-0}
task=${task:-pick_and_lift}

code_dir=$WORK/Code/polarnet/
task_file=${task_file:-$code_dir/polarnet/assets/'10_tasks.csv'}

if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  num_tasks=$(wc -l < $task_file)
  task_id=$(( (${SLURM_ARRAY_TASK_ID} % $num_tasks) + $task_offset ))
  taskvar=$(sed -n "${task_id},${task_id}p" $task_file)
  task=$(echo $taskvar | awk -F ',' '{ print $1 }')
  seed_default=$(( ${SLURM_ARRAY_TASK_ID} / $num_tasks ))
  seed=${seed:-$seed_default}
else
  seed=${seed:-$seed_default}
fi

log_dir=$WORK/logs
data_dir=data/10tasks_data/

mkdir -p $data_dir
mkdir -p $log_dir

module load singularity

img_size=128

image=polarnet.sif

export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:$code_dir"

pushd $code_dir/polarnet
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
	$SINGULARITY_ALLOWED_DIR/${image} \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	/usr/bin/python3.9 preprocess/generate_dataset_microsteps.py \
    --save_path $data_dir/train/microsteps/seed${seed} \
    --all_task_file assets/all_tasks.json \
    --image_size $img_size,$img_size --renderer opengl \
    --episodes_per_task $num_episodes \
    --tasks ${task} --variations 1 --offset 0 \
    --processes 1 --seed ${seed}

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
	$SINGULARITY_ALLOWED_DIR/${image} \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	/usr/bin/python3.9 preprocess/generate_dataset_keysteps.py \
    --microstep_data_dir $data_dir/train/microsteps/seed${seed} \
    --keystep_data_dir $data_dir/train/keysteps/seed${seed} \
    --tasks ${task}

# check the correctness of generated keysteps
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
	$SINGULARITY_ALLOWED_DIR/${image} \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	/usr/bin/python3.9 preprocess/evaluate_dataset_keysteps.py \
    --microstep_data_dir $data_dir/train/microsteps/seed${seed} \
    --keystep_data_dir $data_dir/train/keysteps/seed${seed} \
    --image_size $img_size $img_size \
    --tasks ${task} --headless 

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