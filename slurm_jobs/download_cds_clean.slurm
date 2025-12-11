#!/bin/bash

#SBATCH --job-name=what
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL

# Note: 
# 1. job-name will be overrided by the sbatch command.
# 2. NEW_LOG_DIR_BASE_NAME, LOG_DIR_ROOT and WANDB_API_KEY should be exported in the sbatch command.

###############################################################
#   Set up logging and experiment directory
###############################################################
echo $NEW_LOG_DIR_BASE_NAME
echo $LOG_DIR_ROOT
echo "-----"

exp_home="${NEW_LOG_DIR_BASE_NAME}_${SLURM_JOB_ID}"
mkdir -p $exp_home
log_file=$exp_home/raw.log
git_file=$exp_home/git_info
script_file=$(realpath "$0")
slurm_var_file=$exp_home/slurm_env

touch $log_file
git log -1 --pretty=format:"%cd %H %s" > ${git_file}
cp -v ${script_file} ${exp_home}/$(basename "$0")
env | grep "SLURM_" > ${slurm_var_file}


###############################################################
# Activate environment
###############################################################
module load Anaconda3/2024.06
source activate codonfm

###############################################################
#    Your code goes below this line
###############################################################
cd $MYGROUP/CodonFM/data_scripts/data_curation

mkdir -p $MYGROUP/ncbi_refseq_reference/processed_grouped

wget https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt
wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip
unzip -o taxdmp.zip
rm taxdmp.zip

# will skip download and processing.
python download_cds_clean.py > ${log_file} 2>&1


# After job is done, copy the vanilla slurm log to the experiment directory
set -x
cp -v ${LOG_DIR_ROOT}/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out ${exp_home}/slurm_stdout.log