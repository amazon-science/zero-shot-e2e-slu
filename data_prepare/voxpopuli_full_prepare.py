import os
import jsonlines
from speechbrain.dataio.dataio import read_audio, merge_csvs
from speechbrain.utils.data_utils import download_file
import shutil

from .voxpopuli_prepare import ner2semantics, rename_filter_csv

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)


def prepare_slue_voxpopuli_full(
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
    splits = ["asr_train", "asr_dev", "asr_test"]
    for split in splits:
        if os.path.exists(os.path.join(save_folder, split+'.csv')):
            print(f'csv of {split} already exists')
            continue
        else:
            tsv_df = pd.read_csv(os.path.join(data_folder, split+'.tsv'), delimiter='\t', header=0)
            renamed_tsv_df = tsv_df.rename(columns={'id': 'wav'})

            # remove empty _normalized_text
            remove_row_list = []
            for index, row in renamed_tsv_df.iterrows():
                if pd.isna(row["normalized_text"]) or len(row["normalized_text"]) == 0:
                    remove_row_list.append(index)
            renamed_tsv_df2 = renamed_tsv_df.drop(index=remove_row_list)

            renamed_tsv_df2.insert(0, 'ID', range(0, renamed_tsv_df2.shape[0]))

            refer_slurp_dict = {
                "ID": "ID",
                "wav": "wav",
                "normalized_text":"transcript"
            }

            renamed_tsv_df3 = rename_filter_csv(renamed_tsv_df2, refer_slurp_dict, process_semantics=False)

            renamed_tsv_df3 = pd.DataFrame(renamed_tsv_df3, columns=['ID', 'wav', 'semantics', 'transcript'])


            # renamed_tsv_df2.insert(0, 'ID', range(0, renamed_tsv_df2.shape[0]))

            new_filename = os.path.join(save_folder, split+'.csv')
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            renamed_tsv_df3.to_csv(new_filename, index=False)
            print(f'finished transfer {split} from tsv to csv')
    