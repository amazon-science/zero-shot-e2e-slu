#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
#python python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP.yaml
CUDA_VISIBLE_DEVICES=2 python ./train_s2_by_ratio.py hparams/train_s2_ratio_SLURP_realonly.yaml
# --debug


