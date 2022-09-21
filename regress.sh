#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --job-name=regress

module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh
python prompt_result_regression.py \
    --indir '/scratch/mcinerney.de/metaicl/eval_results/results_gpt2meta6' \
    --outdir '/scratch/mcinerney.de/metaicl/regression_results/results_gpt2meta6' \
    --mode transformer_regression \
    --setting no_example_overlap
#    --run_id 2bs103g0 \
#    --no_training
#    --setting no_training
#    --load_checkpoint '/scratch/mcinerney.de/metaicl/regression_results/results_gpt2meta6/'
#    --mode indicator_regression
#    --mode transformer_regression
#    --mode pairwise_comparator_ranking
#    --from_pretrained roberta-large
#    --setting no_example_overlap
#    --setting no_dev_overlap
