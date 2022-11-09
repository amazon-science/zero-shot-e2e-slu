#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
python ./train_s3.py hparams/train_s3_T_SLURP_A_PEOPLESPEECH.yaml hparams/train_NLU.yaml