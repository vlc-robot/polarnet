#!/bin/bash
#SBATCH --job-name=vlc_eval_val
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

cd ${SLURM_SUBMIT_DIR}
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
export PYTHONPATH=/opt/YARR/

module purge
pwd; hostname; date;

seed=${seed:-0}
taskvar=${task:-pick_and_lift+0}

code_dir=$WORK/Code/polarnet/

log_dir=$WORK/logs/

mkdir -p $log_dir

module load singularity

. $WORK/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH
conda activate polarnet

export PYTHONPATH="$PYTHONPATH:$code_dir"

models_dir=exprs/${taskvar}_model/seed${seed}
instr_embed_file=data/10tasks_data/taskvar_instrs/clip/

init_step=${init_step:-50000}
max_step=${max_step:-100000}
step_jump=10000

pushd $code_dir/polarnet
# validation: select the best epoch
for step in $( eval echo {${init_step}..${max_step}..${step_jump}} )
do
srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
	$SINGULARITY_ALLOWED_DIR/polarnet.sif \
    xvfb-run -a python eval_tst_split.py \
    --exp_config  ${models_dir}/logs/training_config.yaml \
    --seed 100 --num_demos 20 \
    --checkpoint ${models_dir}/ckpts/model_step_${step}.pt \
    --taskvars ${taskvar} \
    --num_workers 1 --instr_embed_file $instr_embed_file
done
popd 
