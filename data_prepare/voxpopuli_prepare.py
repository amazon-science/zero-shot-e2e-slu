import os
import jsonlines
from speechbrain.dataio.dataio import read_audio, merge_csvs
from speechbrain.utils.data_utils import download_file
import shutil

def ner2semantics(ner_list, ner_text):
    res = "[]"
    if ner_list == "None":
        return res
    res = []
    ner_list = eval(ner_list)
    for ele in ner_list:
        mid = {}
        mid['type']= ele[0].lower()
        mid['filler']=ner_text[ele[1]:(ele[1]+ele[2])].lower()
        res.append(mid)
    semantics = str(res).replace(",", "|")
    return semantics

def rename_filter_csv(src_df, refer_dict, process_semantics):
    tar_df = src_df.rename(columns=refer_dict)
    chosen_list = list(refer_dict.values())

    if process_semantics == True:
        tar_df.insert(tar_df.shape[1], "semantics", '-1')
        for index, row in tar_df.iterrows():
            mid = ner2semantics(row['normalized_ner'], row['transcript'])
            tar_df.loc[index, 'semantics'] = mid
        chosen_list.append('semantics')

    tar_df = pd.DataFrame(tar_df, columns=chosen_list)
    print(f"tranfer finished, tar_df has the keys of {tar_df.columns}")
    return tar_df

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)


def prepare_slue_voxpopuli(
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

    splits = ["slue-voxpopuli_fine-tune", "slue-voxpopuli_dev", "slue-voxpopuli_test_blind"]
    for i in range(len(splits)):
        split = splits[i]
        if split == splits[len(splits)-1]:
            break
        if os.path.exists(os.path.join(save_folder, split+'.csv')):
            print(f'csv of {split} already exists')
            continue
        else:
            tsv_df = pd.read_csv(os.path.join(data_folder, split+'.tsv'), delimiter='\t', header=0)
            renamed_tsv_df = tsv_df.rename(columns={'id': 'wav'})

            remove_row_list = []
            for index, row in renamed_tsv_df.iterrows():
                if pd.isna(row["normalized_text"]) or len(row["normalized_text"]) == 0:  # or row['normalized_ner'] == 'None'
                    remove_row_list.append(index)
            renamed_tsv_df2 = renamed_tsv_df.drop(index=remove_row_list)

            renamed_tsv_df2.insert(0, 'ID', range(0, renamed_tsv_df2.shape[0]))

            # merge split into wav
            for index, row in renamed_tsv_df2.iterrows():
                mid = row['split'] + '/' + row['wav']
                renamed_tsv_df2.loc[index, 'wav'] = mid

            refer_slurp_dict = {
                "ID": "ID",
                "wav": "wav",
                "normalized_text":"transcript",
            }

            renamed_tsv_df3 = rename_filter_csv(renamed_tsv_df2, refer_slurp_dict, process_semantics=True)

            renamed_tsv_df3 = pd.DataFrame(renamed_tsv_df3, columns=['ID', 'wav', 'semantics', 'transcript'])


            if split == "slue-voxpopuli_fine-tune":
                new_filename = os.path.join(save_folder, split+'.csv')
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                renamed_tsv_df3.to_csv(new_filename, index=False)
                print(f'finished transfer {split} from tsv to csv')

            elif split == "slue-voxpopuli_dev":
                row_num3 = renamed_tsv_df3.shape[0]
                renamed_tsv_df_dev = renamed_tsv_df3.loc[0:int(row_num3/2), :]
                renamed_tsv_df_test = renamed_tsv_df3.loc[int(row_num3/2):, :]

                dev_file_name = os.path.join(save_folder, split+'.csv')
                renamed_tsv_df_dev.to_csv(dev_file_name, index=False)

                test_file_name = os.path.join(save_folder, splits[len(splits)-1]+'.csv')
                renamed_tsv_df_test.to_csv(test_file_name, index=False)

            else:
                raise ValueError('the split name is wrong')


