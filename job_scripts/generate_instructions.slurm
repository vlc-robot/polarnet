
#!/bin/bash
#SBATCH --job-name=generate_instr
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
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
data_dir=data/
log_dir=logs

mkdir -p $data_dir
mkdir -p $log_dir

module load singularity

export PYTHONPATH="$PYTHONPATH:$code_dir"

pushd $code_dir/polarnet/
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	singularity exec \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,$HOME:$HOME --nv \
	$SINGULARITY_ALLOWED_DIR/polarnet2.sif \
	xvfb-run -a -e $log_dir/$SLURM_JOBID.err \
	/usr/bin/python3.9 preprocess/generate_instructions.py \
    --encoder clip \
    --output_file $data_dir/taskvar_instrs/clip \
	--generate_all_instructions --env_file assets/all_tasks.json
popd