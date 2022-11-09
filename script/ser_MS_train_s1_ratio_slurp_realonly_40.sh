#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
#python python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train_s1_by_ratio.py hparams/ser_train_s1_ratio_SLURP_realonly_40.yaml --data_parallel_backend
# --debug


