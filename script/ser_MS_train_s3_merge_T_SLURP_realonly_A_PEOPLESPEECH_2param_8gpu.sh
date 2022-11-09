dname=$(dirname "$PWD")
cd $dname
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py hparams/train.yaml --data_parallel_backend
#python -m torch.distributed.launch --nproc_per_node=4 ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_A_PEOPLESPEECH.yaml --distributed_launch --distributed_backend='nccl' hparams/train_NLU.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./train_s3_merge_setting.py hparams/ser_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH.yaml --data_parallel_backend hparams/ser_train_NLU_by_t2_slurp.yaml
#python experiment.py params.yaml --data_parallel_backend