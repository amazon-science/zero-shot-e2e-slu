#conda activate base
test_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl"
function_path="/home/jfhe/Documents/MountHe/jfhe/projects/slurp_metrics/scripts/evaluation"

# s1_T2
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19811/slurp/slurp_s1/gb_inference.jsonl"


echo "test_jsonal is ${test_jsonal}"
echo "predict_jsonal is ${predict_jsonal}"
echo "function_path is ${function_path}"

cd ${function_path}
python evaluate.py -g ${test_jsonal} -p ${predict_jsonal}


# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions_te_vs_csv_train.jsonl


