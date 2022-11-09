dname=$(dirname "$PWD")
cd $dname
python ./train_s3.py hparams/train_s3.yaml