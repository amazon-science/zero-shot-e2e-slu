#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH.yaml hparams/train_NLU_by_t2_slurp.yaml
# --debug


