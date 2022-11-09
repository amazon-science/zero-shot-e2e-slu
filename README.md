# Zero-Shot End-to-End Spoken Language Understanding via Cross-Modal Selective Self-Training

Work by Jianfeng He [(@he159ok)](https://github.com/he159ok), Julian Salazar [(@JulianSlzr)](https://github.com/JulianSlzr), Kaisheng Yao, Haoqi Li, and Jason Cai at AWS AI Labs.

Conventional spoken language understanding (SLU) systems remain appealing as they can be built without SLU data. Specifically, an automatic speech recognition (ASR) system and a natural language understanding (NLU) system can be trained with disjoint ASR and NLU datasets, then composed to perform SLU. Hence, we consider the task of zero-shot, end-to-end spoken language understanding, where an end-to-end SLU system is trained without audio-semantics pairs. We propose new benchmarks (MiniPS2SLURP, VoxPopuli2SLUE) and demonstrate a baseline system using self-training. We then focus on the real-world setting of found speech, where a large ASR corpus is collected independently from the NLU task. This corpus is likely dominated by speech whose text is not representative of the NLU task, resulting in imbalanced and noisy semantic pseudo labels. Hence, we propose \textit{cross-modal selective self-training} (CMSST), a sample-efficient adaptively filtered approach using a novel \textit{multi-view clustering-based sample selection} (MCSS) to mitigate imbalance and a novel \textit{cross-modal SelectiveNet} (CMSN) to reduce the impact of noise. Each component also independently improves performance: Entity F1 on VoxPopuli2SLUE increases 3.4\% with MCSS, and MiniPS2SLURP increases 8.7\% with CMSN.
## Configure Enviroment
To configure enviroment, we should make sure that `CUDA>=11.1` and then run below commands,
```
conda env create -f env_speechbrain4.yaml
conda activate speechbrain4
```

## Build Data
All data applied in our project is public. 
Please first build a `dataset` folder in anywhere. Then, we list our four datasets and their organization in our project as below.

1. [SLUE-Voxpopuli](https://github.com/asappresearch/slue-toolkit/blob/main/README.md)

Please install the [slue-toolkit](https://github.com/asappresearch/slue-toolkit/blob/main/README.md) in the `dataset/slue` folder at first. 
Then under the `slue-toolkit` folder, please run  below
```
bash scripts/download_datasets.sh
```
After that, the organization of SLUE-Voxpopuli is `datasets/slue/slue-toolkit/data/slue-voxpopuli/`.

2. [Voxpopuli](https://github.com/facebookresearch/voxpopuli)

Please install the `voxpopuli.git` according the instruction on [Voxpopuli](https://github.com/facebookresearch/voxpopuli).
Then, to get the Voxpoluli-English data, please run below
```
python -m voxpopuli.download_audios --root [ROOT] --subset asr
```
where `[ROOT]` is the folder to save your data. Plus, segment these audios and align them with transcripts via
```
python -m voxpopuli.get_asr_data --root [ROOT] --lang en
```
After that, the organization of Voxpopuli-English is `datasets/slue/voxpopuli/voxpopuli/slue-voxpopuli-full/transcribed_data/en/`.


3. [SLURP](https://github.com/pswietojanski/slurp)

Please first download the textual annotation from its `dataset/slurp/`.
For the corresponding acoustic data, you can download it accroding the instruction from its github, or skip it, which will be downloaded by our project code.
It will be easier to have the same dataset organization to ours by our project code.

After that, the organization of SLURP is `datasets/SLURP/`， where four files (`test.jsonl`, `train_real.jsonl`, `train_synthetic.jsonl` and `devel.jsonl`) and three folders (`slurp_real/`, `slurp_split/` and `slurp_synth/`) are available.

4. [PeolpleSpeech](https://mlcommons.org/en/peoples-speech/)

It is the mini-set of PeopleSpeech, which is no more available from the official website. 
We release its mini-set after consulting the authors of [PeolpleSpeech](https://mlcommons.org/en/peoples-speech/), please download it use the mini-set according to its [official]((https://mlcommons.org/en/peoples-speech/)) license.

After that, the organization of peoplespeech is `datasets/peoplespeech/`, where it has a folder `subset/` and a file `flac_train_manifest.jsonl`.

## Models on SLURP 
### Training of SLU3 (our models)
For SLURP as target data and PeopleSpeech as auxiliary data, we only use LSTM as the encoder of NLU for the further experiments.

**BS.** For `BS` model, we can run below,
```
python CMSST_main.py hparams/initial_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_no_selective.yaml hparams/initial_train_NLU_by_t2_slurp.yaml
```
where `hparams/initial_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_01_01_01_65.yaml` provides the parameters for the data preprare, data processing, and SLU3 training;
where `hparams/initial_train_NLU_by_t2_slurp.yaml` provides the parameters for the LSTM-based NLU model pretraining and loading.
If BERT-based NLU model is preferred, we can replace `hparams/train_NLU_by_t2_slurp.yaml` by `hparams/train_NLU_by_t2_slurp_bert_freeze.yaml`.

**BS+CMSN.** For `BS+CMSN` model, we can run below, 
```
python CMSST_main.py hparams/initial_train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH_selective_01_01_01_65.yaml hparams/initial_train_NLU_by_t2_slurp.yaml
```

Since the sample filter on SLURP mixed with PeopleSpeech achieves around 99% F1-score by setting the maximum text length as 16, we skip `MVRC+BS+CMSN` on the SLURP.

For more commands to train SLU3 models on SLURP, please refer to `script/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH.sh`.

### Training of SLU2 (baseline)
**TTS-based.** For `TTS-based` SLU2 model, we can run below, 
```
python ./train_s2_by_ratio.py hparams/train_s2_ratio_SLURP_realonly.yaml
```

### Training of SLU1 (baseline)
**Direct.** For `Direct` SLU1 model, we can run below,
```
python ./train_s1_by_ratio.py hparams/train_s1_ratio_SLURP_realonly.yaml
```

### Testing of SLU1, SLU2, and SLU3
For testing any results from SLURP dataset, we can run below,
```
test_jsonal="/datasets/SLURP/test.jsonl"
function_path="/slurp_metrics/scripts/evaluation"
cd ${function_path}
predict_jsonal="/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results/{seed}/slurp/slurp_peoplespeech/(nlu_)gb_inference.jsonl"
python evaluate.py -g ${test_jsonal} -p ${predict_jsonal}
```
where `${test_jsonal}` is the ground truth of slurp test set; `${predict_jsonal}` is the path to our predicted jsonl file, in which the `{seed}` is the parameter listed in the first `.yaml` file of the training command.
The `${predict_jsonal}` has all been printed in the end of the training process, please replace it for the testing or directly revising `{seed}`.
If you test the performance of NLU and SLU, it will be `nlu_gb_inference.jsonl` and `gb_inference.jsonl` respectively in the `${predict_jsonal}`.
As for `${function_path}`, it is the path to a folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation), please refer to it for the installation.
When to install [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation), it will be better to create a new conda enviroment to install it.

**Important** Becuase sometimes the prediction is irregular, which is not considered in original implementation of [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).
Please replace its `scripts/evaluation/evaluate.py` in the folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation) by our `eval/evalute.py`.

For more command of evaluating results on SLURP dataset, please refer to `script/evaluate_slurp_T1_T2.sh`


## Models on SLUE-Voxpopuli 
### Training of SLU3 (our models)
For SLUE-Voxpopuli as target data and Voxpopuli-English as auxiliary data, we mainly use BERT as the encoder of NLU for the further experiments.

**BS.** For `BS` model, we can run below,
```
python ./CMSST_main.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_no_selective_no_multi_view.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml
```
where `train_s3_merge_T_SLUE_A_VOXPOPULI_fix98_bertnlu.yaml` provides the parameters for the data preprare, data processing, and SLU3 training;
where `hparams/train_NLU_by_t2_slue_bert_freeze.yaml` provides the parameters for the BERT-based NLU model pretraining and loading.
If LSTM-based NLU model is preferred, we can replace `hparams/train_NLU_by_t2_slue_bert_freeze.yaml` by `hparams/train_NLU_by_t2_slue.yaml`.


**BS+CMSN.** For `BS+CMSN` model, we can run below, 
```
python ./CMSST_main.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_selective_no_multi_view.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml
```

**MVRC+BS+CMSN.** For `MVRC+BS+CMSN` model, we can run below, 
```
python ./CMSST_main.py hparams/initial_train_s3_merge_T_SLUE_A_VOXPOPULI_3viewsfilter_drawviews_selective_balance_1_1_1.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml
```
For more commands to train SLU3 models on SLUE-Voxpopuli, please refer to `script/train_s3_merge_T_SLURP_realonly_A_PEOPLESPEECH.sh`.

For more commands to train SLU3 models on SLURP, please refer to `script/train_s3_merge_T_SLUE_realonly_A_VOXPOPULI_FULL_2param_fixshuffle.sh`.



### Training of SLU1 (baseline)
**Direct.** For `Direct` SLU1 model, we can run below,
```
python ./train_s1_by_ratio.py hparams/train_s1_ratio_slue_realonly.yaml.yaml
```

### Testing of SLU1, SLU2, and SLU3
For testing any results from SLUE-Voxpopuli dataset, we can run below,
```
project_path="/speechbrain/recipes/SLURP/Zero_shot_cross_modal/"
test_jsonal=${project_path}"processed_data/slue-voxpopuli/slue-voxpopuli_test_blind.csv"
function_path="/slurp_metrics/scripts/evaluation"
cd ${function_path}
predict_jsonal=${project_path}"results/298312/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/(nlu_)gb_inference.jsonl"
python evaluate_slue.py -g ${test_jsonal} -p ${predict_jsonal}
```
where each parameter of the testing command is the same as those used in *Testing of SLU1, SLU2, and SLU3 on SLURP*.

**Important** Becuase the original implementation of [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation) does not consider SLUE-Voxpopuli.
Please add by our `eval/evalute_slue.py` to `scripts/evaluation/evaluate_slue.py` in the folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).

For more command of evaluating results on SLUE-Voxpopuli dataset, please refer to `script/evaluate_slue_T1_T2.sh`.

## NLU models on either SLURP or SLUE-Voxpopuli
Since we need a pre-trained NLU model on SLU3 model training process, we introduce how to train, or load a pre-trained NLU model.

SLURP has `hparams/train_NLU_by_t2_slurp.yaml` and `hparams/train_NLU_by_t2_slurp_bert_freeze.yaml` for the parameters of LSTM-based and BERT-based NLU training or loading.
SLUE-Voxpopuli has `hparams/train_NLU_by_t2_slue.yaml` and `hparams/train_NLU_by_t2_slue_bert_freeze.yaml` for the parameters of LSTM-based and BERT-based NLU training or loading.

Regardless of choices of datasets or NLU models, when we need to train a new NLU model, we set `train_nlu: True` in the one of the above four `.ymal` files, and run a command to train SLU3.
The training process of SLU3 will train a NLU automatically. 
However, if we have a pre-trained NLU model and want to fix a NLU model for comparison, we should set `train_nlu: False` in the one of the above four `.ymal` files and run a command to train SLU3.

To choose use bert-based NLU, please set the `use_nlu_bert_enc: True` in the first yaml file, and set `use_nlu_bert_enc: True` in the second yaml file;
Or else that you prefer use a LSTM-based NLU, please set the `use_nlu_bert_enc: False` in the first yaml file, and set `use_nlu_bert_enc: False` in the second yaml file

## Other Tips

The **BERT-based NLU** training in both SLURP and SLUE-Voxpopuli should be frozen, or else the fine-tuned BERT in seq2seq will lead to the same embedding for each sample. 

If you want to fix the cluster results for **ablation study** of `MVRC` or `CMSN`, please add `refer_seed: {seed number}` to the first `.yaml` file of the training command, where `{seed number}` should be an experimental result you want to compare.


