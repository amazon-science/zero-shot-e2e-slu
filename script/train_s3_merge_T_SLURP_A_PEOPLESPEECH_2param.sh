#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_A_PEOPLESPEECH.yaml hparams/train_NLU.yaml
# --debug


