module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
#source activate jpt
source activate metaicl
ssh login-01 -f -N -T -R 8904:localhost:8904
ssh login-01 -f -N -T -R 6063:localhost:6063
export TOKENIZERS_PARALLELISM=false
. set_hf_caches.sh
jupyter notebook --port 8904
