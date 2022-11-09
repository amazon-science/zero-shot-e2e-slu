#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
#python python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./train_s1_by_ratio.py hparams/ser_train_s1_ratio_SLURP_realonly_20.yaml --data_parallel_backend
# --debug


