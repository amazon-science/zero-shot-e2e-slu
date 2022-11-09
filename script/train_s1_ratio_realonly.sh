#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
CUDA_VISIBLE_DEVICES=0 python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP_realonly.yaml
# --debug


