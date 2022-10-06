#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --job-name=eval

module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh


python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
    --ks 16 \
    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
    --method direct \
    --prompt_with_random_examples_not_from_task \
    -s "('k=16_renft_val1', (0, 1), (0, .15), range(30), None, '_14')" \
       "('k=16_renft_val2', (0, 1), (.15, .3), range(30), None, '_14')" \
    --dataset "ai2_arc,hate_speech18,glue-rte,superglue-cb,superglue-copa,tweet_eval-hate,tweet_eval-stance_atheism,tweet_eval-stance_feminist"
#"quarel", "financial_phrasebank", "openbookqa", "codah", "qasc", "glue-mrpc", "dream", "sick", "commonsense_qa", "medical_questions_pairs",
# "quartz-with_knowledge", "poem_sentiment", "quartz-no_knowledge", "glue-wnli", "climate_fever", "ethos-national_origin",
# "ethos-race", "ethos-religion", "ai2_arc", "hate_speech18", "glue-rte", "superglue-cb", "superglue-copa", "tweet_eval-hate",
# "tweet_eval-stance_atheism", "tweet_eval-stance_feminist"
