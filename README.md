## Configure Enviroment
To configure enviroment, we should make sure that `CUDA>=11.1` and then run below commands,

```
conda env create -f env_speechbrain4.yaml
conda activate speechbrain4
```

## Data Split
Belows is the concrete information related to our Tab. 1 in the paper.

Data Split on **MiniPS2SLURP** (*Found* Setting, *Mismatched* Domain).		

| Data Name           | Data Source         | Data Location                                      | Sample Size |
| ------------------- | ------------------- | -------------------------------------------------- | ----------- |
| A→L, t | SLURP | processed_data/slurp/invs-train-type=direct.csv    | 22782       |
| T→L, t | SLURP               | processed_data/slurp/vs-train-type=direct.csv      | 22783       |
| A→T, t | SLURP               | processed_data/slurp/invs-train-type=direct.csv    | 22782       |
| A→T, o | MiniPS              | processed_data/peoplespeech/peoplespeech_train.csv | 32555       |
| A→T    | slurp & MiniPS      | concat A→T, t and A→T, o | 55037       |
| test                | SLURP               | processed_data/slurp/test-type=direct.csv                | 13078       |
| Other usages        | SLURP               | processed_data/slurp/fine-tune-type=direct.csv           | 5063        |

Data Split on **VoxPopuli2SLUE** (*Matched* Setting, *Matched* Domain).

| Data Name            | Data Source    | Data Location                                                           | Sample Size |
| -------------------- | -------------- | ----------------------------------------------------------------------- | ----------- |
| A→L, t  | SLUE-VoxPopuli | processed_data/slue-voxpopuli/slue-voxpopuli_invs-train-type=direct.csv | 2250        |
| T→L, t  | SLUE-VoxPopuli | processed_data/slue-voxpopuli/slue-voxpopuli_vs-train-type=direct.csv   | 2250        |
| A→T, t | SLUE-VoxPopuli | processed_data/slue-voxpopuli/slue-voxpopuli_invs-train-type=direct.csv | 2250        |
| A→T, o  | VoxPopuli      | processed_data/slue-voxpopuli-full/asr_train.csv                        | 182466      |
| A→T     | SLUE-VoxPopuli & VoxPopuli | concat A→T, t and A→T, o                      | 55037       |
| Test                 | SLUE-VoxPopuli | processed_data/slue-voxpopuli/slue-voxpopuli_test_blind.csv             | 877         |
| Other usages         | SLUE-VoxPopuli | processed_data/slue-voxpopuli/slue-voxpopuli_fine-tune-type=direct.csv  | 500         |

**Tips** 
1. The `processed_data` is a foloder with our provided data split. It only includes data split, the transcripts (texts) and semantic labels. For the audios, please follow below "Build Data" Instructions or align to your own audio locations.
2. The `processed_data` should be related the `freeze_folder` in our training zero-shot e2e SLU yaml files and not be changed.
3. The `Other usages` we listed in the aobve two tables are not used in our zero-shot E2E SLU model, where you can use the respective data to achieve other instersting goals.

## Build Data
All data applied in our project is publicly released. 
Please first build a `dataset` folder in anywhere. Then, we list our used four datasets and their organization in our project as below.
The audio should be downloaded by the below instructions.

1. [SLUE-Voxpopuli](https://github.com/asappresearch/slue-toolkit/blob/main/README.md)

Please install the [slue-toolkit](https://github.com/asappresearch/slue-toolkit/blob/main/README.md) in the `dataset/slue` folder at first. 
Then under the `slue-toolkit` folder, please run  below
```
bash scripts/download_datasets.sh
```
After that, the organization of SLUE-Voxpopuli is `datasets/slue/slue-toolkit/data/slue-voxpopuli/`.

2. [Voxpopuli](https://github.com/facebookresearch/voxpopuli)

Please install the `voxpopuli.git` according to the instruction on [Voxpopuli](https://github.com/facebookresearch/voxpopuli).
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
For the corresponding acoustic data, you can download it according the instruction from its Github, or skip it, which will be downloaded by our project code.
It will be easier to have the same dataset organization to ours by our project code.

After that, the organization of SLURP is `datasets/SLURP/`， where four files (`test.jsonl`, `train_real.jsonl`, `train_synthetic.jsonl` and `devel.jsonl`) and three folders (`slurp_real/`, `slurp_split/` and `slurp_synth/`) are available.

4. MinPS

It is the mini-set of [PeopleSpeech]((https://mlcommons.org/en/peoples-speech/)).
We will release its mini-set of [PeolpleSpeech](https://mlcommons.org/en/peoples-speech/) later, please download it according to its [official]((https://mlcommons.org/en/peoples-speech/)) license.

We will release the audio of MiniPS, though they can be obtained by querying the audio names in `wav` column of `processed_data/slue-voxpopuli-full/asr_train.csv`.

After downloading the MiniPS, the organization of peoplespeech is `datasets/peoplespeech/`, where it has a folder `subset/` and a file `flac_train_manifest.jsonl`. 
## Our Models
### Training of our zero-shot E2E SLU model

**MCSS+CMSN.** For `MCSS+CMSN` model, we can run below, 
```
# train our model on VoxPopuli2SLUE by SentBERT
python ./CMSST_main.py hparams/train_zeroe2eslu_slue_voxpopuli_sentbert.yaml hparams/initial_train_NLU_by_t2_slue_bert_freeze.yaml

# train our model on MiniPS2SLURP by SentBERT
python ./CMSST_main.py hparams/train_zeroe2eslu_slurp_minips_sentbert.yaml hparams/initial_train_NLU_by_t2_slurp_bert_freeze.yaml

```
where the first yaml file provides the parameters for the data preprare, data processing, and our zero-shot E2E SLU model training;
where the second yaml file provides the parameters for the BERT-based NLU model pretraining and loading.
If a LSTM-based NLU model is preferred, please replace `initial_train_NLU_by_t2_slue_bert_freeze.yaml` or `hparams/initial_train_NLU_by_t2_slurp_bert_freeze.yaml` by `initial_train_NLU_by_t2_slue.yaml` or `hparams/initial_train_NLU_by_t2_slurp.yaml`, respectively.


### Testing of our zero-shot E2E SLU model
For testing the model performance, we can run below,
```
cd script

bash ./test_minips2slurp.sh

bash ./test_voxpopuli2slue.sh
```
In the `test_minips2slurp.sh` and `test_voxpopuli2slue.sh`,  `${test_jsonal}` is the ground truth of slurp test set; `${predict_jsonal}` is the path to our predicted jsonl file, in which the `{seed}` is the parameter listed in the first `.yaml` file of the training command.
The `${predict_jsonal}` has all been printed in the end of the training process, please replace it for the testing or directly revising `{seed}`.
If you test the performance of NLU and SLU, it will be `nlu_gb_inference.jsonl` and `gb_inference.jsonl` respectively in the `${predict_jsonal}`.
As for `${function_path}`, it is the path to a folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation), please refer to it for the installation.
When to install [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation), it will be better to create a new conda enviroment to install it.

**Tips** Becuase sometimes the prediction is irregular, which is not considered in original implementation of [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).
Please replace its `scripts/evaluation/evaluate.py` in the folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation) by our `eval/evalute.py`.

**Tips** Becuase the original implementation of [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation) does not consider SLUE-Voxpopuli.
Please add by our `eval/evalute_slue.py` to `scripts/evaluation/evaluate_slue.py` in the folder of installed [slurp evaluation toolkit](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).


## NLU models on either SLURP or SLUE-Voxpopuli
Since we need a pre-trained NLU model on our zero-shot E2E SLU model training process, we introduce how to train, or load a pre-trained NLU model.

SLURP has `hparams/train_NLU_by_t2_slurp.yaml` and `hparams/train_NLU_by_t2_slurp_bert_freeze.yaml` for the parameters of LSTM-based and BERT-based NLU training or loading.
SLUE-Voxpopuli has `hparams/train_NLU_by_t2_slue.yaml` and `hparams/train_NLU_by_t2_slue_bert_freeze.yaml` for the parameters of LSTM-based and BERT-based NLU training or loading.

Regardless of choices of datasets or NLU models, when we need to train a new NLU model, we set `train_nlu: True` in the first ymal file, and run a command to train our zero-shot E2E SLU model.
The training process of our zero-shot E2E SLU model will train a NLU automatically. 
However, if we have a pre-trained NLU model and want to fix a NLU model for comparison, we should set `train_nlu: False` in the one of the above four `.ymal` files and run a command to train our zero-shot E2E SLU model.

To choose use bert-based NLU, please set the `use_nlu_bert_enc: True` in the first yaml file, and set `use_nlu_bert_enc: True` in the second yaml file;
Or else if you prefer to use a LSTM-based NLU, please set the `use_nlu_bert_enc: False` in the first yaml file, and set `use_nlu_bert_enc: False` in the second yaml file.

## Other Tips


If you want to fix the cluster results for **ablation study** of `MCSS` or `CMSN`, please add `refer_seed: {seed number}` to the first `.yaml` file of the training command, where `refer_seed: {seed number}` should be the `seed: {seed number}` in the experimental `.yaml` you want to compare.

This operation will copy the folder of `refer_seed: {seed number}` to foldder `seed: {seed number}`, where only the well-trained model folder will be deleted. 

You can set the parameters in the yaml files related to `refer_seed: {seed number}`, so that you can skip some steps.

## To Do Lists
1. We will release the audio of MiniPS (though you can also find them by query the audio names in the `processed_data/slue-voxpopuli-full/asr_train.csv`).
2. We will release our well-trained models.
