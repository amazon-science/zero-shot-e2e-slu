import numpy as np
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
import pandas as pd
import os, shutil
import json

def prepare_data(hparams, data_name, data_role='tar', split=None):
    if data_role == 'tar':
        role_prefix = ""
    elif data_role == 'aux':
        role_prefix = "aux_"
    else:
        print('data_role is invalid!')
        raise ValueError
    if data_name == 'slurp':
        # if data_role!='tar' or split == None:
        from data_prepare.slurp_prepare import prepare_SLURP  # noqa
        run_on_main(
            prepare_SLURP,
            kwargs={
                "data_folder": hparams[role_prefix+"data_folder"],
                "save_folder": hparams[role_prefix+"freeze_folder"],
                "train_splits": hparams[role_prefix+"train_splits"],
                "slu_type": "direct",
                "skip_prep": hparams[role_prefix+"skip_prep"],
            },
        )




    elif data_name == 'slue-voxpopuli':
        from data_prepare.voxpopuli_prepare import prepare_slue_voxpopuli  # noqa
        run_on_main(
            prepare_slue_voxpopuli,
            kwargs={
                "data_folder": hparams[role_prefix+"data_folder"],
                "save_folder": hparams[role_prefix+"freeze_folder"],
                # "train_splits": hparams[role_prefix+"train_splits"],
                "slu_type": "direct",
                "skip_prep": hparams[role_prefix+"skip_prep"],
            },
        )

    elif data_name == 'slue-voxpopuli-full':
        assert data_role == 'aux'
        from data_prepare.voxpopuli_full_prepare import prepare_slue_voxpopuli_full  # noqa
        run_on_main(
            prepare_slue_voxpopuli_full,
            kwargs={
                "data_folder": hparams[role_prefix+"data_folder"],
                "save_folder": hparams[role_prefix+"freeze_folder"],
                # "train_splits": hparams[role_prefix+"train_splits"],
                "slu_type": "direct",
                "skip_prep": hparams[role_prefix+"skip_prep"],
            },
        )

    elif data_name == 'peoplespeech':
        from data_prepare.peoplespeech_prepare import prepare_peoplespeech  # noqa
        run_on_main(
            prepare_peoplespeech,
            kwargs={
                "data_folder": hparams[role_prefix+"data_folder"],
                "save_folder": hparams[role_prefix+"freeze_folder"],
                # "train_splits": hparams[role_prefix+"train_splits"],
                "slu_type": "direct",
                "skip_prep": hparams[role_prefix+"skip_prep"],
            },
        )

    elif data_name == 'librispeech':
        from data_prepare.librispeech_prepare import prepare_librispeech  # noqa
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams[role_prefix+"data_folder"],
                "tr_splits": hparams[role_prefix+"train_splits"],
                "dev_splits": hparams[role_prefix+"dev_splits"],
                "te_splits": hparams[role_prefix+"test_splits"],
                "save_folder": hparams[role_prefix+"freeze_folder"],
                "merge_lst": hparams[role_prefix+"train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams[role_prefix+"skip_prep"],
            },
        )

def process_data(hparams, data_name, data_role='tar', split_ratio="None"):

    if data_name == 'slurp':
        from data_process import slurp_process
        # here we create the datasets objects as well as tokenization and encoding
        if split_ratio!="None" and hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
            (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer) = slurp_process.dataio_prepare_ratio(hparams,
                                                                                                              data_role, split_ratio)
            return (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer)
        elif hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
            (vs_train_data, invs_train_data, valid_data, test_data, tokenizer) = slurp_process.dataio_prepare(hparams, data_role)
            return (vs_train_data, invs_train_data, valid_data, test_data, tokenizer)
        else:
            (train_set, valid_set, test_set, tokenizer) = slurp_process.dataio_prepare(hparams, data_role)
            return (train_set, valid_set, test_set, tokenizer)

    elif data_name == 'librispeech':
        from data_process import librispeech_process

        # here we create the datasets objects as well as tokenization and encoding
        (
            train_data,
            valid_data,
            test_datasets,
            train_bsampler,
            valid_bsampler,
        ) = librispeech_process.dataio_prepare(hparams, data_role)
        return         (
            train_data,
            valid_data,
            test_datasets,
            train_bsampler,
            valid_bsampler,
        )

    elif data_name == 'slue-voxpopuli':
        from data_process import slue_voxpopuli_process

        if split_ratio!="None" and hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli':
            (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer) = slue_voxpopuli_process.dataio_prepare_ratio(hparams,
                                                                                                              data_role, split_ratio)
            return (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer)
        # here we create the datasets objects as well as tokenization and encoding
        elif hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli':

            (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer) = slue_voxpopuli_process.dataio_prepare(hparams, data_role, split_ratio)
            return (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer)
        else:
            (train_set, valid_set, test_set, tokenizer) = slue_voxpopuli_process.dataio_prepare(hparams, data_role)
            return (train_set, valid_set, test_set, tokenizer)

    elif data_name == 'slue-voxpopuli-full':
        from data_process import slue_voxpopuli_full_process

        # here we create the datasets objects as well as tokenization and encoding
        if hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli-full':
            (vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer) = slue_voxpopuli_full_process.dataio_prepare(
                hparams, data_role)
            return (vs_train_data, invs_train_data, valid_data, test_data, tokenizer)
        else:
            (train_set, valid_set, test_set, tokenizer) = slue_voxpopuli_full_process.dataio_prepare(hparams, data_role)
            return (train_set, valid_set, test_set, tokenizer)


    elif data_name == 'peoplespeech':
        from data_process import peoplespeech_process

        # here we create the datasets objects as well as tokenization and encoding
        (train_set, valid_set, test_set, tokenizer) = peoplespeech_process.dataio_prepare(hparams, data_role)
        return (train_set, valid_set, test_set, tokenizer)

def split_csv_ratio(source_name, target_name1, target_name2, target_name3, split_ratio, remain_train_ratio=100, small_set=False):


    source_df = pd.read_csv(source_name, header=0, sep=',')


    ini_row_num, _ = source_df.shape
    remain_row_num = int(ini_row_num * remain_train_ratio / 100)
    if small_set == True:
        remain_row_num = 128
    source_df = source_df.loc[0: remain_row_num, :]


    # shuffled_source_df = source_df.sample(frac=1) # for reproducity, cancel shuffle
    shuffled_source_df = source_df # cancel shuffle in all the experiments

    row_num, col_num = source_df.shape
    split_ratio = split_ratio.split(" ")
    split_id = [0]
    for ele in split_ratio:
        split_id.append(int(row_num * float(ele)))
    # mid_row_num = int(row_num / 2)
    target_names = [target_name1, target_name2, target_name3]
    for i in range(len(split_ratio)):
        mid_shuffled_source_df = shuffled_source_df[split_id[i]:split_id[i+1]]
        mid_shuffled_source_df.to_csv(target_names[i], index=False)
    print(f'{source_name} has been orderly divided into {target_name1}, {target_name2} and {target_name3}')

def split_csv(source_name, target_name1, target_name2):
    if os.path.exists(target_name1) and os.path.exists(target_name2):
        print(f'subsets {target_name1} and {target_name2} are already existing.')
        return
    source_df = pd.read_csv(source_name, header=0, sep=',')
    # key_names = list(source_df.head(0))
    shuffled_source_df = source_df.sample(frac=1)
    row_num, col_num = source_df.shape
    mid_row_num = int(row_num / 2)
    shuffled_source_df_half1 = shuffled_source_df[0:mid_row_num]
    shuffled_source_df_half2 = shuffled_source_df[mid_row_num:]
    shuffled_source_df_half1.to_csv(target_name1, index=False)
    shuffled_source_df_half2.to_csv(target_name2, index=False)
    print(f'{source_name} has been randomly divided into {target_name1} and {target_name2}')

def filter_csv(file_name, remained_id_list, seleted_aux_csv_path):
    source_df = pd.read_csv(file_name, header=0, sep=',')

    res_df = source_df.loc[source_df['ID'].isin(remained_id_list)]

    assert res_df.shape[0] == len(remained_id_list)

    res_df.to_csv(seleted_aux_csv_path, index=False)
    print(f'save the selected aux csv to f{seleted_aux_csv_path}')
    return res_df

def merge_tar_aux_data(tar_source, aux_source, save_name, refer_slurp_dict, tar_name, aux_name):
    tar_df = pd.read_csv(tar_source, header=0, sep=',')
    aux_df = pd.read_csv(aux_source, header=0, sep=',')
    aux_df = aux_df.rename(columns=refer_slurp_dict)
    new_key = 'ori_source'
    extracted_tar_df = tar_df.loc[:, refer_slurp_dict.values()]
    extracted_tar_df.insert(extracted_tar_df.shape[1], new_key, tar_name)

    extracted_aux_df = aux_df.loc[:, refer_slurp_dict.values()]
    extracted_aux_df.insert(extracted_aux_df.shape[1], new_key, aux_name)

    selected_keys = list(refer_slurp_dict.values())
    selected_keys.append(new_key)

    extracted_tar_aux_df = pd.concat([extracted_tar_df, extracted_aux_df], keys=selected_keys)
    extracted_tar_aux_df = extracted_tar_aux_df.rename(columns={'ID':'ori_id'})
    extracted_tar_aux_df = extracted_tar_aux_df.sample(frac=1, random_state=1)  # here is the shuffle

    extracted_tar_aux_df.insert(0, 'ID', range(0, extracted_tar_aux_df.shape[0]))

    extracted_tar_aux_df.to_csv(save_name, index=False)
    print(f'merge is finished and saved to {save_name}')

    # return tar_df.shape[0], aux_df.shape[0]

def merge_tar_aux_data_general(tar_source, aux_source, save_name, tar_refer_slurp_dict, aux_refer_slurp_dict, tar_name, aux_name):
    if aux_refer_slurp_dict == None:
        raise ValueError("at least should select filter range of key values for the selection")
    tar_df = pd.read_csv(tar_source, header=0, sep=',')
    aux_df = pd.read_csv(aux_source, header=0, sep=',')
    if tar_refer_slurp_dict != None:
        tar_df = tar_df.rename(columns=tar_refer_slurp_dict)
    if aux_refer_slurp_dict != None:
        aux_df = aux_df.rename(columns=aux_refer_slurp_dict)
    new_key = 'ori_source'
    extracted_tar_df = tar_df.loc[:, aux_refer_slurp_dict.values()]
    extracted_tar_df.insert(extracted_tar_df.shape[1], new_key, tar_name)

    extracted_aux_df = aux_df.loc[:, aux_refer_slurp_dict.values()]
    extracted_aux_df.insert(extracted_aux_df.shape[1], new_key, aux_name)

    selected_keys = list(aux_refer_slurp_dict.values())
    selected_keys.append(new_key)

    extracted_tar_aux_df = pd.concat([extracted_tar_df, extracted_aux_df], keys=selected_keys)
    if "ID" in list(extracted_tar_aux_df.columns):
        extracted_tar_aux_df = extracted_tar_aux_df.rename(columns={'ID':'ori_id'})
    extracted_tar_aux_df = extracted_tar_aux_df.sample(frac=1, random_state=1)  # here is the shuffle

    extracted_tar_aux_df.insert(0, 'ID', range(0, extracted_tar_aux_df.shape[0]))

    extracted_tar_aux_df.to_csv(save_name, index=False)
    print(f'merge is finished and saved to {save_name}')

    # return tar_df.shape[0], aux_df.shape[0]

def check_tar_aux_data(tar_source, aux_source, save_name):
    tar_df = pd.read_csv(tar_source, header=0, sep=',')
    aux_df = pd.read_csv(aux_source, header=0, sep=',')
    mer_df = pd.read_csv(save_name, header=0, sep=',')
    tar_df_num = tar_df.shape[0]
    aux_df_num = aux_df.shape[0]
    mer_df_num = mer_df.shape[0]
    assert mer_df_num == (tar_df_num + aux_df_num)
    return tar_df_num, aux_df_num , mer_df_num


def insert_lable_slurp(ini_sel_file, syn_save_file, insert_sel_file):
    ini_sel_df = pd.read_csv(ini_sel_file, header=0, sep=',')
    # ini_sel_df.insert(ini_sel_df.shape[1], 'semantics', 0)

    syn_dict = {
        'ID': [],
        'semantics': []
    }
    mid_dict = {}
    with open(syn_save_file, encoding='utf-8') as f:
        sample_lines = f.readlines()
        total_sample = len(sample_lines)
        for i in range(total_sample):
            sample = json.loads(sample_lines[i])

            mid_dict['ID'] = int(sample['ID'])
            mid_dict['semantics'] = {
                'scenario': sample['scenario'],
                'action': sample['action'],
                'entities': sample['entities']
            }
            mid_dict['semantics'] = str(mid_dict['semantics']).replace(",", "|")
            syn_dict['ID'].append(mid_dict['ID'])
            syn_dict['semantics'].append(mid_dict['semantics'])

    # res_df = pd.Series(res_dict)
    syn_df = pd.DataFrame(syn_dict, columns=list(syn_dict.keys()))

    ins_sel_df = pd.merge(ini_sel_df, syn_df, on='ID')
    ins_sel_df.sort_values("ori_id", inplace=True)
    ins_sel_df = ins_sel_df.sample(frac=1, random_state=1)
    ins_sel_df.to_csv(insert_sel_file, index=False)

def insert_lable_slue(ini_sel_file, syn_save_file, insert_sel_file):
    ini_sel_df = pd.read_csv(ini_sel_file, header=0, sep=',')
    # ini_sel_df.insert(ini_sel_df.shape[1], 'semantics', 0)

    syn_dict = {
        'ID': [],
        'semantics': []
    }
    mid_dict = {}
    with open(syn_save_file, encoding='utf-8') as f:
        sample_lines = f.readlines()
        total_sample = len(sample_lines)
        for i in range(total_sample):
            sample = json.loads(sample_lines[i])

            mid_dict['ID'] = int(sample['ID'])
            mid_dict['semantics'] = sample['entities']

            mid_dict['semantics'] = str(mid_dict['semantics']).replace(",", "|")
            syn_dict['ID'].append(mid_dict['ID'])
            syn_dict['semantics'].append(mid_dict['semantics'])


    # res_df = pd.Series(res_dict)
    syn_df = pd.DataFrame(syn_dict, columns=list(syn_dict.keys()))

    ins_sel_df = pd.merge(ini_sel_df, syn_df, on='ID')
    ins_sel_df.sort_values("ori_id", inplace=True)
    ins_sel_df = ins_sel_df.sample(frac=1, random_state=1)
    ins_sel_df.to_csv(insert_sel_file, index=False)

def str_list_find_two_key(str_list, key1, key2):
    for ele in str_list: # should divide and equal find, not in
        ele_str = ele.split('-')
        if len(ele_str) != 2:
            continue
        if (key2 == ele_str[1]) and (key1 in ele_str[0]):
            # print(ele)
            return ele
    return None

def str_list_find_one_key(str_list, key1):
    for ele in str_list:
        if key1 in ele:
            return ele
    return None

def gen_tts_id2audio_dict(tts_split_foder, tts_audio_folder, save_path=None):

    res = {}


    split_folder_list = os.listdir(tts_split_foder)
    split_id_txt_list = []
    split_sentence_txt_list = []
    for ele in split_folder_list:
        if 'id' in ele and 'sentence' in ele:
            raise ValueError('id and sentence keywords appear together')
        if 'id' in ele:
            split_id_txt_list.append(ele)
        elif 'sentence' in ele:
            split_sentence_txt_list.append(ele)

    split_keywords = ['train', 'devel', 'test'] # test is wrong

    audio_folder_list = os.listdir(tts_audio_folder)


    # for id_txt_file in split_id_txt_list:
    for split_keyword in split_keywords:
        id_txt_file = str_list_find_one_key(split_id_txt_list, split_keyword)
        sen_txt_file = str_list_find_one_key(split_sentence_txt_list, split_keyword)

        id_txt_file = os.path.join(tts_split_foder, id_txt_file)
        sen_txt_file = os.path.join(tts_split_foder, sen_txt_file)

        sample_id_f = open(id_txt_file)
        sample_sens_f = open(sen_txt_file)
        sample_ids = sample_id_f.readlines()
        sample_sens = sample_sens_f.readlines()
        jsq = 0
        for sample_id in sample_ids:
            if '\n' in sample_id:
                sample_id = sample_id[:-1]
            audio_folder = str_list_find_two_key(audio_folder_list, split_keyword, str(jsq))
            if sample_id in res.keys():
                raise ValueError("the id is exist")
            if sample_sens[jsq][0:-1] not in res.keys():
                res[sample_sens[jsq][0:-1]] = [audio_folder]
            else:
                res[sample_sens[jsq][0:-1]].append(audio_folder)

            jsq += 1

    np.save(save_path, res)
    print(f'id2ttsaudio_dict has been saved to {save_path}')
    return res


def merge_tts_invs(src_csv, tts_2audio, tar_csv):
    source_df = pd.read_csv(src_csv, header=0, sep=',')
    source_df.insert(source_df.shape[1], 'tts_wav', 0)
    jsq_gen = 0
    jsq_syn = 0
    for index, row in source_df.iterrows():
        # print(jsq)
        jsq_gen += 1
        query = row['transcript']
        if query in tts_2audio.keys():
            ret_res = tts_2audio[query][0]
        elif query[0:-1] in tts_2audio.keys():
            ret_res = tts_2audio[query[0:-1]][0]
        else:
            jsq_syn += 1
            ret_res = row['wav']
        source_df.loc[index, 'tts_wav'] = ret_res
    print(f'{jsq_syn}/{source_df.shape[0]} are unique in the syn')
    assert jsq_syn == 0

    source_df.to_csv(tar_csv)
    print(f'tts_invs_csv_tr is saved to {tar_csv}')


def assign_abstudy_folder(
        main_folder_id,
        refer_folder_id,
        res_folder_path,
        data_name
):
    assert main_folder_id != refer_folder_id

    if data_name == 'slurp':
        remove_path = 'slurp/save/direct'
    elif data_name == 'slue-voxpopuli':
        remove_path = 'slue-voxpopuli/save/direct'
    else:
        raise ValueError('data_name is wrong!')

    main_res_folder_path = os.path.join(res_folder_path, str(main_folder_id))
    refer_res_folder_path = os.path.join(res_folder_path, str(refer_folder_id))
    remove_pretrained_model_folder = os.path.join(main_res_folder_path, remove_path)

    if len(os.listdir(remove_pretrained_model_folder)) != 0:
        print(f'There is a pretrained model in {remove_pretrained_model_folder}')
        raise ValueError('please change "retrain_s3" value or remove the content under the folder')

    if os.path.exists(main_res_folder_path):
        shutil.rmtree(main_res_folder_path)
    shutil.copytree(refer_res_folder_path, main_res_folder_path)
    print(f'copy {refer_res_folder_path} to {main_res_folder_path} finished!' )


    shutil.rmtree(remove_pretrained_model_folder)
    print(f'remove and rebuild {remove_pretrained_model_folder} finished!')
    os.makedirs(remove_pretrained_model_folder)
    if len(os.listdir(remove_pretrained_model_folder)) != 0:
        raise ValueError(f'invalid remove of {remove_pretrained_model_folder}')






