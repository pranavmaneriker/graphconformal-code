#!/bin/bash

SCRIPTDIR="$HOME/AVOIRPlusPlus"
output_dir="$SCRIPTDIR/graphconformal_sweep"
base_output_dir="$SCRIPTDIR/outputs"
ALPHA=0.1

# for DATASET in "CiteSeer" "PubMed" "Cora"; do
for DATASET in "Amazon_Photos" "Amazon_Computers" "Coauthor_CS" "Coauthor_Physics" "Flickr" "ogbn-arxiv" "ogbn-products"; do
	    BASE_JOB_ID=best_${DATASET}_split_0.2_0.2
	    BASE_OUTPUT_DIR=${base_output_dir}/${DATASET}/split/0.2_0.2/

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpuserial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-0:0:0
#SBATCH -J benchmark_${DATASET}
#SBATCH -e logs/benchmark_${DATASET}_%j.err
#SBATCH -o logs/benchmark_${DATASET}_%j.out

echo Job started at `date` on `hostname`
source ~/.bashrc
conda deactivate
conda activate fairgraph

export DGLBACKEND=pytorch

cd ${SCRIPTDIR}
python time_improvements_cfgnn.py --dataset ${DATASET} --base_job_id ${BASE_JOB_ID} --base_output_dir ${BASE_OUTPUT_DIR} --dataset_dir ${SCRIPTDIR}/datasets --conformal_seed 0 --alpha $ALPHA --n_runs_per_expt 5 --benchmark_output_file ${SCRIPTDIR}/analysis/benchmarking/${DATASET}_0.2_0.2.csv
EOT
done
