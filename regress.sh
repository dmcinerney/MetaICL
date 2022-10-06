#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=regress

module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh
python prompt_result_regression.py \
    --task medical_questions_pairs \
    --metric acc_normalized \
    --indir '/scratch/mcinerney.de/metaicl/eval_results/results_gpt2meta7' \
    --outdir '/scratch/mcinerney.de/metaicl/regression_results/results_gpt2meta7' \
    --mode transformer_regression \
    --setting no_example_overlap \
    --no_labels
#    --run_id 2d1ist92
#    --eq_means_lt
#    --lr 1e-7
#    --run_id 35q9e9if \
#    --no_training
#    --setting no_training
#    --load_checkpoint '/scratch/mcinerney.de/metaicl/regression_results/results_gpt2meta6/'
#    --mode indicator_regression
#    --mode transformer_regression
#    --mode pairwise_comparator_ranking
#    --from_pretrained roberta-large
#    --setting no_example_overlap
#    --setting no_dev_overlap
#    --task medical_questions_pairs
#    --task glue-mrpc
