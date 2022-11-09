#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname



if [ $1 == "glove_cluster_100_cos_98_inconsistent_filter_res" ]; # this used unified filter res
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_consistent_filter_res" ]; # this used unified filter res
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_fix98.yaml hparams/train_NLU_by_t2_slue.yaml


elif [ $1 == "glove_cluster_100_cos_98_select_full" ];
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "glove_cluster_100_cos_98_select_only_cm" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_only_cm.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_combo1" ]; # actually it is combo1_75
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_combo1_pen05" ]; # actually it is combo1_75
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1_pen05.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_combo1_pen25" ]; # actually it is combo1_75
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1_pen25.yaml hparams/train_NLU_by_t2_slue.yaml
# next step: need to do pen 002 and sel_weight ab study


elif [ $1 == "glove_cluster_100_cos_98_select_combo1_35" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1_35.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_combo1_55" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1_55.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_combo1_95" ];
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo1_95.yaml hparams/train_NLU_by_t2_slue.yaml


elif [ $1 == "glove_cluster_100_cos_98_select_combo2" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_selective_combo2.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_80_select_3view_sf_drawviews" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "glove_cluster_100_cos_97" ];
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_glove_cluster_100_cos_97.yaml hparams/train_NLU_by_t2_slue.yaml

### below is for bert nlu encoder
elif [ $1 == "glove_cluster_100_cos_98_select_full_bertnlu" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_fix98_bertnlu.yaml hparams/train_NLU_by_t2_slue_bert.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_full_bertnlu_frz" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_fix98_bertnlu.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_smallSF_bertnlu_frz" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=0  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_bert.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_98_select_smallSF_bertnlu_frz_selective" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_bert_selective_combo1.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

#below is the multive sf + select
elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_1_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_0_0" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=0  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_0_0.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_0_1_0" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_0_1_0.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_1_0" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_1_0.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_0_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_0_1.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_0_0_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=0  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_0_0_1.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml
elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_5_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_5_1.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml
elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_5_1_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_5_1_1.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml
elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_balance_1_1_5" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_1_5.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml


elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_random" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=0  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_random.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "glove_cluster_100_cos_80_fix_smallSF_bertnlu_frz_selective_multivewsf_extreme" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_extreme.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml
#above is the multive sf + select


# step1 is random and no step2 and no LC
elif [ $1 == "glove_random_step1_no_step2_no_LC" ];
then
CUDA_VISIBLE_DEVICES=0  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_random_filter_no_sel_no_multiview.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "glove_randomrefer_step1_no_step2_no_LC" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_random_filter_no_sel_no_multiview_refer.yaml hparams/train_NLU_by_t2_slue.yaml


elif [ $1 == "glove_bertnlu_random_step1_no_step2_no_LC" ];
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_nlubert_random_filter_no_sel_no_multiview.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml

# none step-2 none cross modal selective
elif [ $1 == "glove_bertnlu_general_cos_000_no_step1_no_step2_no_LC" ];
then
CUDA_VISIBLE_DEVICES=3  python ./train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_cos_000_no_selective_no_multiviewbalance.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml




##### ini #####
elif [ $1 == "ini_glove_cluster_100_cos_80_fix_bertnlu_frz_selective_multivewsf_balance_1_1_1" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_1_1.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "ini_glove_cluster_100_cos_80_fix_bertnlu_frz_no_selective_no_multivewsf" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_no_selective_no_multi_view.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml

elif [ $1 == "ini_glove_cluster_100_cos_80_fix_bertnlu_frz_selective_no_multivewsf" ]; # frz->freeze
then
CUDA_VISIBLE_DEVICES=2  python ./train_s3_merge_setting.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_selective_no_multi_view.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml


### below is parameter analysis
elif [ $1 == "slue098_lstm_selective_combo1_01_01_001_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_01_01_001_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue098_lstm_selective_combo1_01_01_1_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_01_01_1_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "slue098_lstm_selective_combo1_01_1_01_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_01_1_01_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue098_lstm_selective_combo1_01_001_01_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_01_001_01_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "slue098_lstm_selective_combo1_1_01_01_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_1_01_01_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue098_lstm_selective_combo1_001_01_01_75" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_lstm_selective_001_01_01_75_no_balance.yaml hparams/train_NLU_by_t2_slue.yaml



### above is parameter analysis
elif [ $1 == "slue080_nlubert_no_selective_but_multiview_1_1_1" ]; # 01_01_01 -> 1_1_1 slue080_nlubert_no_selective_but_multiview_01_01_01
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_balance_1_1_1_no_sel.yaml hparams/train_NLU_by_t2_slue_bert_freeze.yaml


elif [ $1 == "slue080_lstm_selective_01_01_01_75_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_01_01_85_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_85_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_01_01_65_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_65_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_01_01_55_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_55_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "slue080_lstm_selective_01_01_01_35_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_35_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_01_01_45_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_45_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_01_01_95_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_01_95_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml

elif [ $1 == "slue080_lstm_selective_01_01_05_55_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_01_05_55_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_01_05_01_55_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_01_05_01_55_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml
elif [ $1 == "slue080_lstm_selective_05_01_01_55_multiview_1_1_1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_05_01_01_55_by_lstm.yaml hparams/train_NLU_by_t2_slue.yaml



##### ini #####


fi
# --debug



