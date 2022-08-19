module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
#source activate jpt
source activate metaicl
ssh login-01 -f -N -T -R 8902:localhost:8902
ssh login-01 -f -N -T -R 6060:localhost:6060
export TOKENIZERS_PARALLELISM=false
jupyter notebook --port 8902
