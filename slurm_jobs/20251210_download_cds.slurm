#!/bin/bash

#SBATCH --job-name=what
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL


###############################################################
base_dir="${MYSCRATCH}/general_exp"
prefix="${SLURM_JOB_NAME}_"
script_path=$(realpath "$0")

exp_home="${MYSCRATCH}/general_exp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p $exp_home
logfile=$exp_home/raw.log
touch $logfile


git log -1 --pretty=format:"%cd %H %s" > ${logfile}
cp ${script_path} ${exp_home}


# module load python/3.11.6 py-pip/23.1.2-py3.11.6
# if ncbi-genome-download not found, install it by:
# pip install ncbi-genome-download

cd $MYGROUP/CodonFM/data_scripts/data_curation

mkdir -p $MYGROUP/ncbi_refseq_reference

wget https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt
wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip
unzip taxdmp.zip
rm taxdmp.zip

python download_cds_clean.py > ${logfile} 2>&1