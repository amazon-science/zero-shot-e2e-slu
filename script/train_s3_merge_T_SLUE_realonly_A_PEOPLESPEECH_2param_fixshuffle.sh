#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI.yaml hparams/train_NLU_by_t2_slue.yaml
# --debug



