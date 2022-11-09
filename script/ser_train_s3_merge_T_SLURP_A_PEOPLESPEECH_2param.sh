#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
python ./train_s3_merge_setting.py hparams/ser_train_s3_merge_T_SLURP_A_PEOPLESPEECH.yaml hparams/ser_train_NLU.yaml
# --debug


