module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (0, .1), range(250), None, '_1')" \
#       "('train', (0, .5), (.1, .2), range(250, 500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.2, .3), range(500, 750), None, '_2')" \
#       "('train', (0, .5), (.3, .4), range(750, 1000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.4, .5), range(1000, 1250), None, '_3')" \
#       "('train', (0, .5), (.5, .6), range(1250, 1500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.6, .7), range(1500, 1750), None, '_4')" \
#       "('train', (0, .5), (.7, .8), range(1750, 2000), None, '_4')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('val', (.5, .75), (.8, .9), range(2000, 2100), None, '_5')" \
#       "('test', (.75, 1), (.9, 1), range(2100, 2200), None, '_5')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (0, .1), range(3000, 3250), None, '_1')" \
#       "('train', (0, .5), (.1, .2), range(3250, 3500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.2, .3), range(3500, 3750), None, '_2')" \
#       "('train', (0, .5), (.3, .4), range(3750, 4000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.4, .5), range(4000, 4250), None, '_3')" \
#       "('train', (0, .5), (.5, .6), range(4250, 4500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.6, .7), range(4500, 4750), None, '_4')" \
#       "('train', (0, .5), (.7, .8), range(4750, 5000), None, '_4')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('val', (.5, .75), (.8, .9), range(5000, 5500), None, '_5')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('test', (.75, 1), (.9, 1), range(5500, 6000), None, '_5')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (0, .1), range(7000, 7250), None, '_1')" \
#       "('train', (0, .5), (.1, .2), range(7250, 7500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.2, .3), range(7500, 7750), None, '_2')" \
#       "('train', (0, .5), (.3, .4), range(7750, 8000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.4, .5), range(8000, 8250), None, '_3')" \
#       "('train', (0, .5), (.5, .6), range(8250, 8500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.6, .7), range(8500, 8750), None, '_4')" \
#       "('train', (0, .5), (.7, .8), range(8750, 9000), None, '_4')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (0, .1), range(10000, 10250), None, '_1')" \
#       "('train', (0, .5), (.1, .2), range(10250, 10500), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.2, .3), range(10500, 10750), None, '_2')" \
#       "('train', (0, .5), (.3, .4), range(10750, 11000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.4, .5), range(11000, 11250), None, '_3')" \
#       "('train', (0, .5), (.5, .6), range(11250, 11500), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.6, .7), range(11500, 11750), None, '_4')" \
#       "('train', (0, .5), (.7, .8), range(11750, 12000), None, '_4')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (0, .1), range(20000, 20500), None, '_1')" \
#       "('train', (0, .5), (.1, .2), range(20500, 21000), None, '_1')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.2, .3), range(21000, 21500), None, '_2')" \
#       "('train', (0, .5), (.3, .4), range(21500, 22000), None, '_2')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.4, .5), range(22000, 22500), None, '_3')" \
#       "('train', (0, .5), (.5, .6), range(22500, 23000), None, '_3')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('train', (0, .5), (.6, .7), range(23000, 23500), None, '_4')" \
#       "('train', (0, .5), (.7, .8), range(23500, 24000), None, '_4')"

#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('val_same_prompts', (0, .5), (.8, .9), range(250), None, '_6')"
#python evaluate_prompts.py --out_dir eval_results/results_gpt2meta6 \
#    -s "('val_same_prompts', (0, .5), (.9, 1), range(250), None, '_6')"
