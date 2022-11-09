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

elif [ $1 == "s3_T1_04_small" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198661/slurp/slurp_s1/gb_inference.jsonl"
elif [ $1 == "s3_T_random_04_small" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198663/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s2_T2_04_small" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198662/slurp/slurp_s2/gb_inference.jsonl"



elif [ $1 == "direct_TR" ]
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/direct/results/19860/predictions.jsonl"

elif [ $1 == "s2_nlu_bert_performance" ]
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198610/slurp/slurp_peoplespeech/nlu_gb_inference.jsonl"

elif [ $1 == "s3_T1_selective_combo1" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198621/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_selective_combo1_95" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198625/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_selective_combo1_85" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198624/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_selective_combo1_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198623/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_selective_combo1_1_01_01_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198626/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_selective_combo1_1_01_01_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198626/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_selective_combo1_001_01_01_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198627/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_selective_combo1_01_001_01_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198629/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_selective_combo1_01_1_01_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198628/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_selective_combo1_01_01_001_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198630/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_selective_combo1_01_01_1_65" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198631/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_random_step1_no_sel_no_multiview" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198632/slurp/slurp_peoplespeech/gb_inference.jsonl"




elif [ $1 == "s3_T1_cluster098_step1_no_sel_no_multiview_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198634/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster090_step1_no_sel_no_multiview_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198635/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster080_step1_no_sel_no_multiview_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/？/slurp/slurp_peoplespeech/gb_inference.jsonl"


elif [ $1 == "s3_T1_cluster095_step1_no_sel_no_multiview_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198636/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_65_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198637/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_45_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198639/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_25_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198638/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_75_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198640/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_85_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198641/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster095_step1_selective_01_01_01_95_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198642/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_cluster080_step1_selective_01_01_01_65_with_multiviewbal_but_bertnlu" ]; # 198644
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198648/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster090_step1_selective_01_01_01_65_with_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198643/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_cluster080_step1_selective_01_01_01_65_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198647/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster090_step1_selective_01_01_01_65_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198646/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_cluster000_no_step1_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198650/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster000_04_05_no_step1_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198658/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster000_04_07_no_step1_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198659/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster000_04_10_no_step1_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198660/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "slurp_people_cluter80_nlubert_nosel_butbalance" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198664/slurp/slurp_peoplespeech/gb_inference.jsonl"






elif [ $1 == "s3_T1_cluster080_step1_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198651/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_textlen16_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198652/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_textlen16_us_selective_1_1_1_065__no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198653/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_textlen16_us_selective_1_1_1_075__no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198657/slurp/slurp_peoplespeech/gb_inference.jsonl"




elif [ $1 == "s3_T1_random_no_selective_no_multiviewbal_but_bertnlu" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198654/slurp/slurp_peoplespeech/gb_inference.jsonl"

elif [ $1 == "s3_T1_cluster_80_no_selective_no_multiviewbal_use_lstm" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198655/slurp/slurp_peoplespeech/gb_inference.jsonl"
elif [ $1 == "s3_T1_cluster_80_use_selective_1_1_1_065_no_multiviewbal_use_lstm" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198656/slurp/slurp_peoplespeech/gb_inference.jsonl"
















elif [ $1 == "s3_T1_selective_combo2" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/198622/slurp/slurp_peoplespeech/gb_inference.jsonl"





fi




echo "test_jsonal is ${test_jsonal}"
echo "predict_jsonal is ${predict_jsonal}"
echo "function_path is ${function_path}"

cd ${function_path}
python evaluate.py -g ${test_jsonal} -p ${predict_jsonal}


# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions_te_vs_csv_train.jsonl


