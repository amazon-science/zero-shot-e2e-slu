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

    ## skip download at first
    # # If the data folders do not exist, we need to download/extract the data
    # if not os.path.isdir(os.path.join(data_folder, "slurp_synth")):
    #     # Check for zip file and download if it doesn't exist
    #     zip_location = os.path.join(data_folder, "slurp_synth.tar.gz")
    #     if not os.path.exists(zip_location):
    #         url = "https://zenodo.org/record/4274930/files/slurp_synth.tar.gz?download=1"
    #         download_file(url, zip_location, unpack=True)
    #     else:
    #         print("Extracting slurp_synth...")
    #         shutil.unpack_archive(zip_location, data_folder)
    #
    # if not os.path.isdir(os.path.join(data_folder, "slurp_real")):
    #     # Check for zip file and download if it doesn't exist
    #     zip_location = os.path.join(data_folder, "slurp_real.tar.gz")
    #     if not os.path.exists(zip_location):
    #         url = "https://zenodo.org/record/4274930/files/slurp_real.tar.gz?download=1"
    #         download_file(url, zip_location, unpack=True)
    #     else:
    #         print("Extracting slurp_real...")
    #         shutil.unpack_archive(zip_location, data_folder)
    #
    # splits = [
    #     "train_real",
    #     "train_synthetic",
    #     "devel",
    #     "test",
    # ]
    # id = 0
    # for split in splits:
    #     new_filename = (
    #         os.path.join(save_folder, split) + "-type=%s.csv" % slu_type
    #     )
    #     if os.path.exists(new_filename):
    #         continue
    #     print("Preparing %s..." % new_filename)
    #
    #     IDs = []
    #     duration = []
    #
    #     wav = []
    #     wav_format = []
    #     wav_opts = []
    #
    #     semantics = []
    #     semantics_format = []
    #     semantics_opts = []
    #
    #     transcript = []
    #     transcript_format = []
    #     transcript_opts = []
    #
    #     jsonl_path = os.path.join(data_folder, split + ".jsonl")
    #     if not os.path.isfile(jsonl_path):
    #         if split == "train_real":
    #             url_split = "train"
    #         else:
    #             url_split = split
    #         url = (
    #             "https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/"
    #             + url_split
    #             + ".jsonl"
    #         )
    #         download_file(url, jsonl_path, unpack=False)


    ## tsv file can be read directly
    # with jsonlines.open(jsonl_path) as reader:
    #     for obj in reader:
    #         scenario = obj["scenario"]
    #         action = obj["action"]
    #         sentence_annotation = obj["sentence_annotation"]
    #         num_entities = sentence_annotation.count("[")
    #         entities = []
    #         for slot in range(num_entities):
    #             type = (
    #                 sentence_annotation.split("[")[slot + 1]
    #                 .split("]")[0]
    #                 .split(":")[0]
    #                 .strip()
    #             )
    #             filler = (
    #                 sentence_annotation.split("[")[slot + 1]
    #                 .split("]")[0]
    #                 .split(":")[1]
    #                 .strip()
    #             )
    #             entities.append({"type": type, "filler": filler})
    #         for recording in obj["recordings"]:
    #             IDs.append(id)
    #             if "synthetic" in split:
    #                 audio_folder = "slurp_synth/"
    #             else:
    #                 audio_folder = "slurp_real/"
    #             path = os.path.join(
    #                 data_folder, audio_folder, recording["file"]
    #             )
    #             signal = read_audio(path)
    #             duration.append(signal.shape[0] / 16000)
    #
    #             wav.append(path)
    #             wav_format.append("flac")
    #             wav_opts.append(None)
    #
    #             transcript_ = obj["sentence"]
    #             if slu_type == "decoupled":
    #                 transcript_ = transcript_.upper()
    #             transcript.append(transcript_)
    #             transcript_format.append("string")
    #             transcript_opts.append(None)
    #
    #             semantics_dict = {
    #                 "scenario": scenario,
    #                 "action": action,
    #                 "entities": entities,
    #             }
    #             semantics_ = str(semantics_dict).replace(
    #                 ",", "|"
    #             )  # Commas in dict will make using csv files tricky; replace with pipe.
    #             semantics.append(semantics_)
    #             semantics_format.append("string")
    #             semantics_opts.append(None)
    #             id += 1
    #
    #     df = pd.DataFrame(
    #         {
    #             "ID": IDs,
    #             "duration": duration,
    #             "wav": wav,
    #             "semantics": semantics,
    #             "transcript": transcript,
    #         }
    #     )
    #     df.to_csv(new_filename, index=False)
    #
    # # Merge train splits
    # train_splits = [split + "-type=%s.csv" % slu_type for split in train_splits]
    # merge_csvs(save_folder, train_splits, "train-type=%s.csv" % slu_type)
