#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP.yaml
# --debug


