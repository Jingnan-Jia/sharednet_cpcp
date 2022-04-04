#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH -e results/logs/slurm-%j.err
#SBATCH -o results/logs/slurm-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com


eval "$(conda shell.bash hook)"

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/logs

##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
# git will not detect the current file because this file may be changed when this job was run
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh


ssh -tt jjia@nodelogin02 /bin/bash << ENDSSH
echo "Hello, I an in nodelogin02 to do some git operations."
cd data/sharednet
git add -A
git commit -m "jobid is ${job_id}"
git push origin master
exit
ENDSSH

echo "Hello, I am back in $(hostname) to run the code"

# shellcheck disable=SC2046
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_${idx}_err.txt 1>${slurm_dir}/slurm-${job_id}_${idx}_out.txt --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --jobid=${job_id} --model_names="pancreas" --cond_flag=True --cond_pos='input' --amp=True --remark="base32"




