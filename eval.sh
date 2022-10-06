#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --job-name=eval

module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh


#python -m pdb evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 0 \
#    --random_preds \
#    -s "('rp_val1', (0, 1), (0, .15), range(5), None, '_1')" \
#       "('rp_val2', (0, 1), (.15, .3), range(5), None, '_1')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 0 \
#    --checkpoints checkpoints/multitask-zero/hr_to_lr/model.pt \
#    --method direct \
#    -s "('zs_val1', (0, 1), (0, .15), range(1), None, '_2')" \
#       "('zs_val2', (0, 1), (.15, .3), range(1), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/metaicl/hr_to_lr/model.pt \
#    --method direct \
#    -s "('k=16_val1', (0, 1), (0, .15), range(30), None, '_3')" \
#       "('k=16_val2', (0, 1), (.15, .3), range(30), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/metaicl/hr_to_lr/model.pt \
#    --method direct \
#    --prompt_with_random_examples_not_from_task \
#    -s "('k=16_renft_val1', (0, 1), (0, .15), range(30), None, '_4')" \
#       "('k=16_renft_val2', (0, 1), (.15, .3), range(30), None, '_4')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/metaicl/hr_to_lr/model.pt \
#    --method direct \
#    --prompt_with_random_wrong_task \
#    -s "('k=16_rwt_val1', (0, 1), (0, .15), range(30), None, '_5')" \
#       "('k=16_rwt_val2', (0, 1), (.15, .3), range(30), None, '_5')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/metaicl/hr_to_lr/model.pt \
#    --method direct \
#    --random_prompt_labels \
#    -s "('k=16_rpl_val1', (0, 1), (0, .15), range(30), None, '_6')" \
#       "('k=16_rpl_val2', (0, 1), (.15, .3), range(30), None, '_6')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 0 \
#    --checkpoints checkpoints/channel-multitask-zero/hr_to_lr/model.pt \
#    --method channel \
#    -s "('zs_val1', (0, 1), (0, .15), range(1), None, '_7')" \
#       "('zs_val2', (0, 1), (.15, .3), range(1), None, '_7')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/channel-metaicl/hr_to_lr/model.pt \
#    --method channel \
#    -s "('k=16_val1', (0, 1), (0, .15), range(30), None, '_8')" \
#       "('k=16_val2', (0, 1), (.15, .3), range(30), None, '_8')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/channel-metaicl/hr_to_lr/model.pt \
#    --method channel \
#    --prompt_with_random_examples_not_from_task \
#    -s "('k=16_renft_val1', (0, 1), (0, .15), range(30), None, '_9')" \
#       "('k=16_renft_val2', (0, 1), (.15, .3), range(30), None, '_9')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/channel-metaicl/hr_to_lr/model.pt \
#    --method channel \
#    --prompt_with_random_wrong_task \
#    -s "('k=16_rwt_val1', (0, 1), (0, .15), range(30), None, '_10')" \
#       "('k=16_rwt_val2', (0, 1), (.15, .3), range(30), None, '_10')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints checkpoints/channel-metaicl/hr_to_lr/model.pt \
#    --method channel \
#    --random_prompt_labels \
#    -s "('k=16_rpl_val1', (0, 1), (0, .15), range(30), None, '_11')" \
#       "('k=16_rpl_val2', (0, 1), (.15, .3), range(30), None, '_11')"




#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 0 \
#    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
#    --method direct \
#    -s "('zs_val1', (0, 1), (0, .15), range(1), None, '_12')" \
#       "('zs_val2', (0, 1), (.15, .3), range(1), None, '_12')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
#    --method direct \
#    -s "('k=16_val1', (0, 1), (0, .15), range(30), None, '_13')" \
#       "('k=16_val2', (0, 1), (.15, .3), range(30), None, '_13')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
#    --method direct \
#    --prompt_with_random_examples_not_from_task \
#    -s "('k=16_renft_val1', (0, 1), (0, .15), range(30), None, '_14')" \
#       "('k=16_renft_val2', (0, 1), (.15, .3), range(30), None, '_14')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
#    --method direct \
#    --prompt_with_random_wrong_task \
#    -s "('k=16_rwt_val1', (0, 1), (0, .15), range(30), None, '_15')" \
#       "('k=16_rwt_val2', (0, 1), (.15, .3), range(30), None, '_15')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2_zs_and_meta \
#    --ks 16 \
#    --checkpoints gpt-j-6B --gpt2s gpt-j-6B --test_batch_size 4 \
#    --method direct \
#    --random_prompt_labels \
#    -s "('k=16_rpl_val1', (0, 1), (0, .15), range(30), None, '_16')" \
#       "('k=16_rpl_val2', (0, 1), (.15, .3), range(30), None, '_16')"







#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc --ks 0 \
#    -s "('train_zs', (0, .5), (0, .2), range(1), None, '_zeroshot')" \
#       "('val_same_prompts_zs', (0, .5), (.8, 1), range(1), None, '_zeroshot')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (0, .2), range(500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.2, .4), range(500, 1000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.4, .6), range(1000, 1500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.6, .8), range(1500, 2000), None, '_4')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('val', (.5, .75), (.8, 1), range(2000, 2500), None, '_5')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('val_same_prompts', (0, .5), (.8, 1), range(500), None, '_6')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (0, .2), range(10000, 10500), None, '_7')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.2, .4), range(10500, 11000), None, '_8')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.4, .6), range(11000, 11500), None, '_9')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.6, .8), range(11500, 12000), None, '_10')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('val', (.5, .75), (.8, 1), range(12000, 12500), None, '_11')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (0, .2), range(20000, 20500), None, '_12')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.2, .4), range(20500, 21000), None, '_13')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.4, .6), range(21000, 21500), None, '_14')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.6, .8), range(21500, 22000), None, '_15')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('val', (.5, .75), (.8, 1), range(22000, 22500), None, '_16')"


#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (0, .2), range(30000, 30500), None, '_17')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.2, .4), range(30500, 31000), None, '_18')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.4, .6), range(31000, 31500), None, '_19')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta7 --dataset glue-mrpc \
#    -s "('train', (0, .5), (.6, .8), range(31500, 32000), None, '_20')"
