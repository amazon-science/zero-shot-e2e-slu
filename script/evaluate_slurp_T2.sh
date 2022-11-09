#conda activate base
test_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl"
function_path="/home/jfhe/Documents/MountHe/jfhe/projects/slurp_metrics/scripts/evaluation"


if [ $1 == "s1_T1" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19811/slurp/slurp_s1/gb_inference.jsonl"
elif [ $1 == "s1_T1_80" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19813/slurp/slurp_s1/gb_inference.jsonl"
elif [ $1 == "s2_T2" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19821/slurp/slurp_s2/gb_inference.jsonl"
elif [ $1 == "s3_T1" ];
then
# non-fixed
# predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19861/slurp/slurp_peoplespeech/gb_inference.jsonl"
# fixed
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19862/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "direct_TR" ]
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/direct/results/19860/predictions.jsonl"
fi


echo "test_jsonal is ${test_jsonal}"
echo "predict_jsonal is ${predict_jsonal}"
echo "function_path is ${function_path}"

cd ${function_path}
python evaluate.py -g ${test_jsonal} -p ${predict_jsonal}


# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions_te_vs_csv_train.jsonl


