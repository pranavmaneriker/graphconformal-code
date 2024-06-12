#!/bin/bash

PROJECTDIR="graphconformal-code/"
CONDAENV=$condaenvname
SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

best_run_dir="${SCRIPTDIR}/../configs/custom_configs/best_base_configs"

#for DATASET in "CiteSeer" "PubMed" "Cora"; do
#for DATASET in "Amazon_Photos" "Amazon_Computers"; do
for DATASET in "Coauthor_CS" "Coauthor_Physics"; do
    for train_frac in 0.2 0.3; do
        for val_frac in 0.1 0.2; do
            best_param_path="${best_run_dir}/${DATASET}/split/${train_frac}_${val_frac}/basegnn_config.yaml"
            if [ ! -f $best_param_path ]; then
                echo "Best parameter file not found for ${DATASET} with train_frac=${train_frac} and val_frac=${val_frac}"
                continue
            fi
            config_output_dir="${best_run_dir}/${DATASET}/split/${train_frac}_${val_frac}"
            mkdir -p ${config_output_dir}
            job_id="best_${DATASET}_split_${train_frac}_${val_frac}"
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -p a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH -J best_${DATASET}
#SBATCH -e logs/best_${DATASET}_%j.err
#SBATCH -o logs/best_${DATASET}_%j.out

echo Job started at `date` on `hostname`
# CONDA SETUP
#. /home/$USER/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch

cd ~/${PROJECTDIR}
python train_base_gnn.py --config_path=${best_param_path} --logging_config.use_wandb False --output_dir ${config_output_dir} --dataset_dir ${SCRIPTDIR}/../datasets --job_id ${job_id} 
EOT
        done
    done
done
