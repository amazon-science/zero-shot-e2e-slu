import os
import jsonlines
from speechbrain.dataio.dataio import read_audio, merge_csvs
from speechbrain.utils.data_utils import download_file
import shutil
import json
import pandas as pd

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)


def prepare_peoplespeech(
    data_folder, save_folder, slu_type, skip_prep=False
):
    """
    This function prepares the SLURP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to SLURP dataset.
    save_folder: path where to save the csv manifest files.
    slu_type : one of the following:

      "direct":{input=audio, output=semantics}
      "multistage":{input=audio, output=semantics} (using ASR transcripts in the middle)
      "decoupled":{input=transcript, output=semantics} (using ground-truth transcripts)

    train_splits : list of splits to be joined to form train .csv
    skip_prep: If True, data preparation is skipped.
    """
    # if skip_prep:
    #     return
    # split = ["slue-voxpopuli_fine-tune.tsv", "slue-voxpopuli_dev.tsv", "slue-voxpopuli_test_blind.tsv"]
    if skip_prep:
        return
    splits = ["peoplespeech_train", "peoplespeech_dev", "peoplespeech_test"]
    ratio = [0.0, 0.7, 0.85, 1.0]
    split_jsq = 0
    jsq = 0
    for split in splits:
        if os.path.exists(os.path.join(save_folder, split+'.csv')):
            print(f'csv of {split} already exists')
            continue
        else:
            # read json file
            sample_sentence_lists = []
            sample_dataset_ids = []
            input_file_name = os.path.join(data_folder, 'flac_train_manifest.jsonl')
            # df = pd.read_json(input_file_name)
            res_dict = {}
            with open(input_file_name, encoding='utf-8') as f:
                sample_lines = f.readlines()
                total_sample = len(sample_lines)
                for i in range(int(total_sample*ratio[split_jsq]), int(total_sample*ratio[split_jsq+1])):
                    sample = json.loads(sample_lines[i])
                    if i in [int(total_sample*ratio[0]), int(total_sample*ratio[1]), int(total_sample*ratio[2])]:
                        res_dict['ID'] = []
                        for key in sample.keys():
                            res_dict[key] = []

                    # print(i)
                    for key in sample.keys():
                        # print(res_dict.keys())
                        # print(sample.keys())
                        res_dict[key].append(sample[key])
                    res_dict['ID'].append(str(i))
                    jsq += 1
            # res_df = pd.Series(res_dict)
            res_df = pd.DataFrame(res_dict, columns=list(res_dict.keys()))

            new_filename = os.path.join(save_folder, split+'.csv')
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            res_df.to_csv(new_filename, index=False)
            print(f'finished transfer {split} from jsonl to csv')
            split_jsq += 1


