#!/bin/bash

PROJECTDIR="$HOME/AVOIRPlusPlus"
CONDAENV=fairgraph
best_run_dir="$PROJECTDIR/outputs"

for DATASET in "Flickr"
do
    for STYLE in "split"
    do
	    for L_TYPES in "GCN" "GAT" "GraphSAGE"
	    do
		    for TRAIN_FRACS in 0.3
		    do
			    for VAL_FRACS in 0.2
			    do
					job_id="hpt_cfgnn_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}"
					best_base_path="${best_run_dir}/${DATASET}/split/${TRAIN_FRACS}_${VAL_FRACS}/${DATASET}/best_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}"
					if [ ! -d "$best_base_path" ]; then
						echo "Best base run not found for ${DATASET} with train_frac=${TRAIN_FRACS} and val_frac=${VAL_FRACS}"
						continue
					fi

sbatch <<EOT
#!/bin/bash
#SBATCH --exclusive

#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=2-0:0:0
#SBATCH -J ${job_id}
#SBATCH -o /jobs/slurm-out-%A.txt
#SBATCH -e /jobs/slurm-err-%A.txt

# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fairgraph/lib:$LD_LIBRARY_PATH

cd ${PROJECTDIR}

srun python hpt_conf_gnn.py --config_path="configs/hpt_conf_gnn_default_flickr.yaml" --dataset ${DATASET} --base_model_dir ${best_base_path} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]" --tune_split_config.val_fracs "[${VAL_FRACS}]"
EOT
				done
			done
		done
	done
done