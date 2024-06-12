#!/bin/bash

for DATASET in "Coauthor_Physics" #"Amazon_Photos" "Amazon_Computers" 
do
    for STYLE in "split" #"n_samples_per_class"
    do
    
SCRIPTDIR="graphconformal-code/"

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-0:0:0

. $HOME/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $condaenvname

export DGLBACKEND=pytorch

cd ~/${SCRIPTDIR}

srun python hpt_base_gnn.py  --config_path="configs/hpt_base_gnn_default.yaml" --dataset ${DATASET} --tune_split_config.s_type ${STYLE}
EOT
        # num_submit_jobs=$(squeue -u $USER -h |  wc -l)
        # while [ $num_submit_jobs -ge 8 ]; do
        #     echo "Waiting for available slots..."
        #     sleep 180  # Wait 
        #     #num_running_jobs=$(squeue -u $USER -h | grep " R " | wc -l)
        #     num_submit_jobs=$(squeue -u $USER -h |  wc -l)
        # done

    done
done