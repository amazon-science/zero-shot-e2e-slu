#conda activate speechbrain4
dname=$(dirname "$PWD")
cd $dname


### below is for bert nlu encoder
if [ $1 == "slurp_people_textlen_nlubert" ];
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_nlu_bert.yaml hparams/train_NLU_by_t2_slurp_bert.yaml

elif [ $1 == "slurp_people_textlen_nlubert_frz" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_nlu_bert.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1.yaml hparams/train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_95" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_95.yaml hparams/train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_85" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_85.yaml hparams/train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_65.yaml hparams/train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_1_01_01_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_1_01_01_65.yaml hparams/train_NLU_by_t2_slurp.yaml
elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_001_01_01_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_001_01_01_65.yaml hparams/train_NLU_by_t2_slurp.yaml


elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_01_001_01_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_01_001_01_65.yaml hparams/train_NLU_by_t2_slurp.yaml
elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_01_1_01_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_01_1_01_65.yaml hparams/train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_01_01_001_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_01_01_001_65.yaml hparams/train_NLU_by_t2_slurp.yaml
elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo1_01_01_1_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_01_01_1_65.yaml hparams/train_NLU_by_t2_slurp.yaml



elif [ $1 == "slurp_people_textlen_nlulstm_selective_combo2" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo2.yaml hparams/train_NLU_by_t2_slurp.yaml

# use step1 random, no step2 sf and no lc
elif [ $1 == "slurp_people_random_sf_no_sel_no_multiview" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_random_filter_no_sel_no_multivew.yaml hparams/train_NLU_by_t2_slurp.yaml


### ini ###
elif [ $1 == "slurp_people_selective" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/initial_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_01_01_01_65.yaml hparams/initial_train_NLU_by_t2_slurp.yaml

elif [ $1 == "slurp_people_no_selective" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/initial_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_no_selective.yaml hparams/initial_train_NLU_by_t2_slurp.yaml
### ini ###


### use text fea_similarity ###
elif [ $1 == "slurp_people_cluter100_step098_step2_none_bertnlu_selective_none" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine098_no_selective_no_multiview.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step090_step2_none_bertnlu_selective_none" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine090_no_selective_no_multiview.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step095_step2_none_bertnlu_selective_none" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine095_no_selective_no_multiview.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step000_step2_none_selective_none_but_bertnlu" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine000_no_selective_no_multiview.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml


elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_65_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_65_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_45_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_45_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_25_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_25_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml

elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_75_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_75_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_85_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_85_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step095_bertnlu_selective_01_01_01_95_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_095_01_01_01_95_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml

elif [ $1 == "slurp_people_cluter100_step090_bertnlu_selective_01_01_01_65_with_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_090_01_01_01_65_with_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step080_bertnlu_selective_01_01_01_65_with_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_080_01_01_01_65_with_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml

elif [ $1 == "slurp_people_cluter100_step090_bertnlu_selective_01_01_01_65_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_090_01_01_01_65_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step080_bertnlu_selective_01_01_01_65_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_step1_080_01_01_01_65_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml


elif [ $1 == "slurp_people_cluter100_step08_step2_1_1_1_bertnlu_selective_01_01_01_65" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_combo1_01_01_01_65_multiview_balance.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step080_bertnlu_no_selective_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_step1_080_no_selectivenet_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml

elif [ $1 == "slurp_people_cluter100_step_textlen16_bertnlu_no_selective_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_step1_textlen16_bert_no_selectivenet_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step_textlen16_bertnlu_selective_01_01_01_65_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_textlen16_bert_01_01_01_65_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step_random_bertnlu_no_selective_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_random_filter_bert_no_sel_no_multivew.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step_textlen16_bertnlu_selective_01_01_01_75_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_textlen16_bert_01_01_01_75_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml


### full special
elif [ $1 == "slurp_people_cluter100_step000_04_05_step2_none_selective_none_but_bertnlu" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine000_no_selective_no_multiview_special04_05.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step000_04_07_step2_none_selective_none_but_bertnlu" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=2  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine000_no_selective_no_multiview_special04_07.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step000_04_10_step2_none_selective_none_but_bertnlu" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine000_no_selective_no_multiview_special04_10.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml
elif [ $1 == "slurp_people_cluter100_step000_04_10_random12580_step2_none_selective_none_but_bertnlu" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_basic_cosine000_no_sel_no_multiview_special04_10_random_12580.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml



### above is full special




elif [ $1 == "slurp_people_cluter100_step080_lstm_no_selective_no_multiviewbal" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=0  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_step1_080_lstm_no_selectivenet_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp.yaml




elif [ $1 == "slurp_people_cluter100_step08_lstm_selective_01_01_01_65_no_multiew" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_step1_080_lstm_selective_01_01_01_65_no_multiviewbal.yaml hparams/train_NLU_by_t2_slurp.yaml


elif [ $1 == "slurp_people_cluter80_nlubert_nosel_butbalance" ]; # frz -> frzee
then
CUDA_VISIBLE_DEVICES=3  python train_s3_merge_setting.py hparams/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_step1_080_with_multiviewbal_but_no_sel.yaml hparams/train_NLU_by_t2_slurp_bert_freeze.yaml








### use text fea_similarity ###

fi
# --debug



