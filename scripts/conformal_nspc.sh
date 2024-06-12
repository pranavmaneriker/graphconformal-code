#!/bin/bash

SCRIPTDIR="graphconformal-code/"
output_dir="/home/$USER/graphconformal_sweep"
best_run_dir="/home/$USER/configs/custom_configs/best_base_configs"
mkdir -p ${output_dir}

ALPHA=0.1
# ALPHA=0.2

# for DATASET in "Amazon_Photos" "Coauthor_CS" "Coauthor_Physics"; do
for DATASET in "CiteSeer" "PubMed" "Cora"; do
	for nspc in 10 20 40 80; do
            best_base_path="${best_run_dir}/${DATASET}/n_samples_per_class/${nspc}/${DATASET}/best_${DATASET}_nspc_${nspc}"
            if [ ! -d "$best_base_path" ]; then
                echo "Best base run not found for ${DATASET} with nspc ${nspc} does not exist"
                continue
            fi
            output_path="${output_dir}/${DATASET}/nspc/${nspc}"
            mkdir -p $output_path
            cd ~/${SCRIPTDIR}/scripts
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -p a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH -J c_${DATASET}_${ALPHA}
#SBATCH -e logs/c_${DATASET}_${ALPHA}_%j.err
#SBATCH -o logs/c_${DATASET}_${ALPHA}_%j.out

echo Job started at `date` on `hostname`
. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $condaenvname

export DGLBACKEND=pytorch

cd ~/${SCRIPTDIR}
python conformal_trials.py --expt_configs_dir ./configs/custom_configs/conformal_trial_configs/ --trials_per_config 50 --base_model_dir ${best_base_path} --results_output_dir ${output_path} --alpha ${ALPHA}
EOT
    done
done
