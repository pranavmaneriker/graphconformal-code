#!/bin/bash

PROJECTDIR="$HOME/graphconformal-code"
CONDAENV=$condaenvname
best_run_dir="$PROJECTDIR/outputs"

for DATASET in "ogbn-products" #"Flickr"
do
    for STYLE in "n_samples_per_class" #"split" # 
    do
	    for L_TYPES in "GCN" #"GAT" #"GraphSAGE" #
	    do
		    # for TRAIN_FRACS in 0.2 0.3
		    # do
			#     for VAL_FRACS in 0.1 0.2
			#     do
			# 		job_id="hpt_cfgnn_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}"
			# 		best_base_path="${best_run_dir}/${DATASET}/split/${TRAIN_FRACS}_${VAL_FRACS}/${DATASET}/best_${DATASET}_split_${TRAIN_FRACS}_${VAL_FRACS}"
			# 		if [ ! -d "$best_base_path" ]; then
			# 			echo "Best base run not found for ${DATASET} with train_frac=${TRAIN_FRACS} and val_frac=${VAL_FRACS}"
			# 			continue
			# 		fi
		    for nspc in 10 20 40 80
		    do
			job_id="hpt_cfgnn_${DATASET}_nspc_${nspc}"
			best_base_path="${best_run_dir}/${DATASET}/n_samples_per_class/${nspc}/${DATASET}/best_${DATASET}_nspc_${nspc}"
			if [ ! -d "$best_base_path" ]; then
				echo "Best base run not found for ${DATASET} with nspc ${nspc} does not exist"
				continue
			fi
			

sbatch <<EOT
#!/bin/bash
#SBATCH --exclusive

#SBATCH --partition=gpuserial-40core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=2-0:0:0
#SBATCH -J ${job_id}

# CONDA SETUP
source ~/.bashrc
conda deactivate
conda activate ${CONDAENV}

export DGLBACKEND=pytorch
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fairgraph/lib:$LD_LIBRARY_PATH

cd ${PROJECTDIR}

# srun python hpt_conf_gnn.py --config_path="configs/hpt_conf_gnn_default.yaml" --dataset ${DATASET} --base_model_dir ${best_base_path} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]" --tune_split_config.val_fracs "[${VAL_FRACS}]"
srun python hpt_conf_gnn.py --config_path="configs/hpt_conf_gnn_default.yaml" --dataset ${DATASET} --base_model_dir ${best_base_path} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.samples_per_class "[${nspc}]"
EOT
				# done
			done
		done
	done
done