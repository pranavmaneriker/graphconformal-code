#!/bin/bash
DATASET="PubMed"
SCRIPTDIR="graphconformal-code/"
num_submit_jobs=0

sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH -J tune_${DATASET}
#SBATCH -e logs/tune_${DATASET}_%j.err
#SBATCH -o logs/tune_${DATASET}_%j.out

echo Job started at `date` on
source ~/.bashrc
conda activate $CONDAENV
cd ~/${SCRIPTDIR}

python hpt_base_gnn.py --config_path="configs/custom_configs/hpt_base_small.yaml" --dataset ${DATASET}
EOT
