dname=$(dirname "$PWD")
cd $dname
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py hparams/train.yaml --data_parallel_backend
#python -m torch.distributed.launch --nproc_per_node=4 ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_A_PEOPLESPEECH.yaml --distributed_launch --distributed_backend='nccl' hparams/train_NLU.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_A_PEOPLESPEECH.yaml --data_parallel_backend hparams/train_NLU.yaml
#python experiment.py params.yaml --data_parallel_backend