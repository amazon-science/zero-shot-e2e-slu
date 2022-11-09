import os
import numpy as np
import json
from tqdm.contrib import tqdm

import sys
sys.path.append('..')

def text_len_filter(hparams, train_data, merge_data):
    # hparams: parameters in the main file
    # train_data: should set as T2 data, and can only use its text info
    # merge_data: T1 + A data, and can only use its text inf0

    ## 3.1 if the pre-calculated representations do not exist,
    ## calculate text representation for texts in both target dataset and auxiliary dataset

    print("in text_len_filter, skip the representations for train_data and aux_train_data")

    filter_text_len = hparams['limited_text_len']

    ini_filter_res = {}
    num_cluster = 1 # for text_len type, has no cluster, so set it as 1

    tar_sam_num = 0
    aux_sam_num = 0
    filter_tar_sam_num = 0
    filter_aux_sam_num = 0

    for i in range(num_cluster):
        ini_filter_res[str(i)] = {}
        ini_filter_res[str(i)]['ids'] = []
        ini_filter_res[str(i)]['scores'] = []
        ini_filter_res[str(i)]['ori_source'] = []



    total_aux_samples = merge_data.__len__()


    if hparams["cal_filter_res_by_text_similary"]:
        id_key_name, text_key_name, ori_source_key_name = 'id', 'transcript', 'ori_source'
        jsq = 0


        # should have batch_size for 1 here
        with tqdm(merge_data, initial=0, dynamic_ncols=True, disable=False) as t:
            for batch in t:
                jsq += 1
                if jsq % 10000 == 0:
                    print(f'finish filter by text len of {jsq}/{total_aux_samples}')

                for i in range(num_cluster):
                    if batch[ori_source_key_name] == hparams['data_name']:
                        tar_sam_num += 1
                    elif batch[ori_source_key_name] == hparams['aux_data_name']:
                        aux_sam_num += 1
                    else:
                        raise ValueError('The value of key ori_source in merge_data is not consistent')


                    sample_text_len = len(batch[text_key_name].split())
                    if sample_text_len < filter_text_len:
                        ini_filter_res[str(i)]['ids'].append(batch[id_key_name])
                        ini_filter_res[str(i)]['scores'].append(sample_text_len)
                        ini_filter_res[str(i)]['ori_source'].append(batch[ori_source_key_name])
                        if batch[ori_source_key_name] == hparams['data_name']:
                            filter_tar_sam_num += 1
                        elif batch[ori_source_key_name] == hparams['aux_data_name']:
                            filter_aux_sam_num += 1

                        break
        if not os.path.isdir(hparams['filter_res_by_text_similary_dir']):
            os.makedirs(hparams['filter_res_by_text_similary_dir'])
        np.save(hparams['ini_text_filter_res'], ini_filter_res)
        print("initial_filter_res has been finished at, ", hparams['ini_text_filter_res'])

    else:

        ini_filter_res = np.load(hparams['ini_text_filter_res'], allow_pickle=True).item()


    # data analysis
    if (filter_tar_sam_num + filter_aux_sam_num) == 0 or tar_sam_num == 0:
        print('filter_tar_sam_num + filter_aux_sam_num or tar_sam_num = 0, cannot cal F1 stat')

    else:
        precision = filter_tar_sam_num / (filter_tar_sam_num + filter_aux_sam_num)
        recall = filter_tar_sam_num / (tar_sam_num)
        f1_score = 2 * precision * recall / (precision + recall)
        print(f'in text_len_upper limitation {filter_text_len}, precision is {precision}, recall is {recall}, f1 is {f1_score}')
        print(f'total sample is, {filter_tar_sam_num+filter_aux_sam_num}')

    print('the nums of aux data to each cluster sentence are below')
    print('target_dataset, ', hparams['data_name'])
    print('aux_dataset, ', hparams['aux_data_name'])
    print('filter text length upper limitation, ', filter_text_len)
    print('the total rest data is: ', np.array([len(ini_filter_res[str(x)]['ids']) for x in range(num_cluster)]).sum())
    return ini_filter_res, num_cluster
