#!/bin/bash

for DATASET in "Cora" #"PubMed" #"Flickr"
do
    for STYLE in "n_samples_per_class" #"split" #
    do
	    for L_TYPES in "GAT" "GCN" "GraphSAGE"
	    do
		    #for TRAIN_FRACS in 0.2 0.3
		    #do
		#	    for VAL_FRACS in 0.1 0.2
		#	    do
		    for spc in 10 20 40
		    do
    
				SCRIPTDIR="AVOIRPlusPlus/"

sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1-0:00:00
#SBATCH -p a100-long
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH -J tune_${DATASET}

. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $condaenvname

export DGLBACKEND=pytorch

cd ~/${SCRIPTDIR}
#srun python hpt_base_gnn.py --config_path="configs/custom_configs/hpt_base_gnn_default.yaml" --dataset ${DATASET} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]" --tune_split_config.val_fracs "[${VAL_FRACS}]"
srun python hpt_base_gnn.py --config_path="configs/custom_configs/hpt_base_gnn_default.yaml" --dataset ${DATASET} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.samples_per_class "[${spc}]"
#srun python hpt_base_gnn.py --config_path="configs/custom_configs/hpt_base_gnn_default.yaml" --dataset ${DATASET} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]'

EOT
		#	num_submit_jobs=$(squeue -u $USER -h |  wc -l)
		#	while [ $num_submit_jobs -ge 6 ]; do
		#	    echo "Waiting for available slots..."
		#	    sleep 180  # Wait 
		#	    #num_running_jobs=$(squeue -u $USER -h | grep " R " | wc -l)
		#	    num_submit_jobs=$(squeue -u $USER -h |  wc -l)
		#	done
			#done
		done
	done
    done
done
#srun python hpt_base_gnn.py --config_path="configs/custom_configs/hpt_base_gnn_default.yaml" --dataset ${DATASET} --tune_split_config.s_type ${STYLE} --l_types '["${L_TYPES}"]' --tune_split_config.train_fracs "[${TRAIN_FRACS}]"
