#conda activate base
project_path="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/"
test_jsonal=${project_path}"processed_data/slue-voxpopuli/slue-voxpopuli_test_blind.csv"
function_path="/home/jfhe/Documents/MountHe/jfhe/projects/slurp_metrics/scripts/evaluation"


if [ $1 == "s1_T1" ];
then
predict_jsonal=${project_path}"results/2981/slue-voxpopuli/slue-voxpopuli_s1/gb_inference.jsonl"

elif [ $1 == "s3_T1_random_step1_only" ];
then
predict_jsonal=${project_path}"results/298338/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "s3_T1_random2_step1_only" ];
then
predict_jsonal=${project_path}"results/298342/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "s3_T1_random3_step1_only" ];
then
predict_jsonal=${project_path}"results/298343/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"



elif [ $1 == "s3_nlu_T2" ];
then
predict_jsonal=${project_path}"results/2986/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/nlu_inference_syn_label.jsonl"

elif [ $1 == "nlu_T2_non_bertenc" ];
then
predict_jsonal=${project_path}"results/29835/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/nlu_gb_inference.jsonl"
#predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29835/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/nlu_gb_inference.jsonl"

elif [ $1 == "nlu_T2_bertenc_frz" ];
then
predict_jsonal=${project_path}"results/298312/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/nlu_gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz" ];
then
predict_jsonal=${project_path}"results/298314/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_random_step1_only_bertenc_frz" ];
then
predict_jsonal=${project_path}"results/298339/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"


elif [ $1 == "s3_T1_bertenc_frz_selective" ];
then
predict_jsonal=${project_path}"results/298315/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_1_1" ];
then
predict_jsonal=${project_path}"results/298316/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_0_1_0" ];
then
predict_jsonal=${project_path}"results/298321/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_0_0" ];
then
predict_jsonal=${project_path}"results/298320/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_1_0" ];
then
predict_jsonal=${project_path}"results/298333/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_0_1" ];
then
predict_jsonal=${project_path}"results/298332/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_0_0_1" ];
then
predict_jsonal=${project_path}"results/298334/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_5_1" ];
then
predict_jsonal=${project_path}"results/298335/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_5_1_1" ];
then
predict_jsonal=${project_path}"results/298336/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_balance_1_1_5" ];
then
#predict_jsonal=${project_path}"results/298337/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"  # this might be wrong due to truncate in merge process
predict_jsonal=${project_path}"results/298340/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"








elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_random" ];
then
predict_jsonal=${project_path}"results/298319/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_bertenc_frz_selective_multiview_sf_extreme" ];
then
predict_jsonal=${project_path}"results/298318/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"








#elif [ $1 == "s1_T1_80" ];
#then
#predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19813/slurp/slurp_s1/gb_inference.jsonl"
#elif [ $1 == "s2_T2" ];
#then
#predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/19821/slurp/slurp_s2/gb_inference.jsonl"
elif [ $1 == "s3_T1_0.98" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/2986/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
#elif [ $1 == "direct_TR" ]
#then
#predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/direct/results/19860/predictions.jsonl"
elif [ $1 == "s3_T1_0.98_basic_ab_study_clusters" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29835/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_1_1_1_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29831/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_0_1_0_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29832/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_01_01_01_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29833/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_01_01_01_35" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29838/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_01_01_01_55" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29836/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_01_01_01_95" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29837/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"




elif [ $1 == "s3_T1_0.98_selective_0_01_0_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29834/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_0_01_05_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298310/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "s3_T1_0.98_selective_0_01_25_75" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298311/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"


elif [ $1 == "s3_T1_0.000001_bert_no_selection_no_multiview" ]; # use all VoxPopuli data by
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298341/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"


### below is parameter analysis
elif [ $1 == "slue098_lstm_selective_combo1_01_01_001_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298345/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue098_lstm_selective_combo1_01_01_1_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298344/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "slue098_lstm_selective_combo1_01_1_01_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298346/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue098_lstm_selective_combo1_01_001_01_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298348/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

elif [ $1 == "slue098_lstm_selective_combo1_1_01_01_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298349/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue098_lstm_selective_combo1_001_01_01_75" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298350/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"



elif [ $1 == "slue080_lstm_selective_01_01_01_75_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298351/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_55_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298354/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_85_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298353/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_65_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298352/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_35_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298357/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_45_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298356/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
elif [ $1 == "slue080_lstm_selective_01_01_01_95_multiview_1_1_1" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298355/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"






elif [ $1 == "slue080_nlubert_no_selective_but_multiview_01_01_01" ]; # frz -> frzee
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/298347/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"




elif [ $1 == "s3_T1_0.97" ];
then
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/29830/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"
fi







echo "test_jsonal is ${test_jsonal}"
echo "predict_jsonal is ${predict_jsonal}"
echo "function_path is ${function_path}"

cd ${function_path}
python evaluate_slue.py -g ${test_jsonal} -p ${predict_jsonal}


# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/test.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions.jsonl

# python evaluate.py -g /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/datasets/SLURP/train_real.jsonl -p /home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/NLU/results/better_tokenizer/198621/predictions_te_vs_csv_train.jsonl


