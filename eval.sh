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
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (0, .2), range(500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.2, .4), range(500, 1000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.4, .6), range(1000, 1500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.6, .8), range(1500, 2000), None, '_4')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('val', (.5, .75), (.8, 1), range(2000, 2500), None, '_5')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('val_same_prompts', (0, .5), (.8, 1), range(500), None, '_6')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (0, .2), range(10000, 10500), None, '_7')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.2, .4), range(10500, 11000), None, '_8')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.4, .6), range(11000, 11500), None, '_9')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
#    -s "('train', (0, .5), (.6, .8), range(11500, 12000), None, '_10')"
python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset medical_questions_pairs \
    -s "('val', (.5, .75), (.8, 1), range(12000, 12500), None, '_11')"
