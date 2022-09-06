module load anaconda3/3.7
module load cuda/10.2
module load gcc/5.5.0
source activate metaicl
. set_hf_caches.sh
#python -m torch.distributed.launch --nproc_per_node=3 evaluate_prompts.py
#python evaluate_prompts.py
python -m pdb evaluate_prompts.py
