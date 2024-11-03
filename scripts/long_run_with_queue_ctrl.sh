#!/bin/bash
DATASET="Cora"
SCRIPTDIR="graphconformal-code/"
num_submit_jobs=0

for train_frac in 0.1 0.2 0.3 0.4; do
    for val_frac in 0.1 0.2; do
        for calib_frac in 0.1 0.2 0.3; do
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-04:00:00
#SBATCH -p a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH -J graph_conformal_${DATASET}
#SBATCH -e logs/${DATASET}_%j.err
#SBATCH -o logs/${DATASET}_%j.out

echo Job started at `date` on
conda activate $CONDAENV
cd ~/${SCRIPTDIR}

srun python main.py --epochs 100 --dataset ${DATASET} --num_workers 16 --logging_config configs/custom_configs/logging_config.json --confgnn_args_file configs/custom_configs/confgnn_config.json --dataset_train_frac ${train_frac} --dataset_val_frac ${val_frac} --dataset_calib_frac ${calib_frac}
EOT
        num_submit_jobs=$(squeue -u $USER -h |  wc -l)
        while [ $num_submit_jobs -ge 5 ]; do
            echo "Waiting for available slots..."
            sleep 180  # Wait 
            #num_running_jobs=$(squeue -u $USER -h | grep " R " | wc -l)
            num_submit_jobs=$(squeue -u $USER -h |  wc -l)
        done

        done
    done
done
