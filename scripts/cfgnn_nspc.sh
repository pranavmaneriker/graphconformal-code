#!/bin/bash


SCRIPTDIR="$HOME/graphconformal-code"
output_dir="$SCRIPTDIR/graphconformal_sweep"
base_output_dir="$SCRIPTDIR/outputs"
best_configs_dir="$SCRIPTDIR/configs/custom_configs/best_cfgnn_configs"
mkdir -p ${output_dir}

CONDAENV=fairgraph

for ALPHA in 0.1 0.2 0.3 0.4; do
for DATASET in "Amazon_Computers" "Amazon_Photos"; do
	for nspc in 10 20 40 80; do
            best_base_path="${base_output_dir}/${DATASET}/n_samples_per_class/${nspc}"
            best_cfgnn_path="${best_configs_dir}/${DATASET}/n_samples_per_class/${nspc}/cfgnn_config.yaml"
            if [ ! -f "$best_cfgnn_path" ]; then
                echo $best_cfgnn_path
                echo "Best cfgnn run not found for ${DATASET} with nspc ${nspc} does not exist"
                continue
            fi
            output_path="${output_dir}/${DATASET}/nspc/${nspc}"
            mkdir -p $output_path
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=2-0:0:0
#SBATCH -J best_${DATASET}
#SBATCH -e logs/c_${DATASET}_${ALPHA}_%j.err
#SBATCH -o logs/c_${DATASET}_${ALPHA}_%j.out

echo Job started at `date` on `hostname`
source ~/.bashrc
conda deactivate
conda activate $CONDAENV

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
python run_conformal.py --config_path=${best_cfgnn_path} --logging_config.use_wandb False --output_dir ${best_base_path} --results_output_dir ${output_path} --alpha ${ALPHA} --epochs 100 --conformal_method cfgnn
EOT
    done
done
done
