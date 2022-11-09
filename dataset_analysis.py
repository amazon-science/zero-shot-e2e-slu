# This is the function to analyze the dataset basic properties.

import sys

import matplotlib.pyplot as plt
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd

from data_process import librispeech_process, slurp_process, merge_process
import prepare_process_data
import prepare_aux_text_embedding
import os

import select_aux_sample

import torchtext.vocab as vocab
import numpy as np
import data_process
from text_filter import word2vec_filter

from models import NLU_text2sem, direct_audio2sem

import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator

def analyze_csv(source_file, key_words):
    source_df = pd.read_csv(source_file, header=0, sep=',')
    res = {}
    for key_word in key_words:
        # if key_word in ['duration']:
        #     mid_list = source_df[key_word]
        #     mid_arr = np.array(mid_list)


        if key_word in ['transcript', 'text', 'normalized_text']:
            ini_list = source_df[key_word]
            try:
                mid_list = [len(str(x).split(' ')) for x in ini_list]
            except:
                print(ini_list[0])
                print(key_word)
                print(source_file)
                mid_list = [len(x.split(' ')) for x in ini_list]
                assert 1 == 0

            mid_arr = np.array(mid_list)


        res[key_word] = {}
        res[key_word]['mean'] = mid_arr.mean()
        res[key_word]['std'] = mid_arr.std()
        res[key_word]['min'] = mid_arr.min()
        res[key_word]['max'] = mid_arr.max()
        res[key_word]['info'] = mid_arr
    return res
    print('finished')



if __name__ == "__main__":
    data_names_list = [
        ['slurp', 'peoplespeech'],
        ['slurp', 'slue-voxpopuli'],
        ['slurp', 'slue-voxpopuli-full'],
        ['slue-voxpopuli', 'peoplespeech'],
        ['slue-voxpopuli', 'slurp'],
        ['slue-voxpopuli', 'slue-voxpopuli-full']
    ]
    row_num = 1
    col_num = len(data_names_list)
    res = plt.figure(figsize=(27, 3.5))
    jsq = 0
    for data_names in data_names_list:
        # data_names = ['slurp', 'peoplespeech']

        data_stat = {}
        project_folder = "/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/processed_data"
        # aux_pro_folder = os.path.join(project_folder, 'processed_data')
        for data_name in data_names:
            data_stat[data_name] = {}
            if data_name == 'slurp':
                file_path = data_name+"/train-type=direct.csv"
                data_stat[data_name]['file_path'] = os.path.join(project_folder, file_path)
                data_stat[data_name]['file_key_words'] = ['transcript'] # ['duration', 'transcript']
            elif data_name == 'peoplespeech':
                file_path = data_name+"/peoplespeech_train.csv"
                data_stat[data_name]['file_path'] = os.path.join(project_folder, file_path)
                data_stat[data_name]['file_key_words'] = ['text'] # ['duration', 'text']
            elif data_name == 'slue-voxpopuli':

                file_path = data_name + "/slue-voxpopuli_fine-tune.csv"
                data_stat[data_name]['file_path'] = os.path.join(project_folder, file_path)
                data_stat[data_name]['file_key_words'] = ['normalized_text']
            elif data_name == 'slue-voxpopuli-full':

                file_path = data_name + "/asr_train.csv"
                data_stat[data_name]['file_path'] = os.path.join(project_folder, file_path)
                data_stat[data_name]['file_key_words'] = ['normalized_text']
            else:
                print('data_name is set wrongly')
                raise ValueError


        ### 1. evn_build
        # hparams_file = sys.argv[1]
        # with open(hparams_file) as tar_fin:
        #     hparams = load_hyperpyyaml(tar_fin)  # include the processes to load modules



        tar_data_tr_file = data_stat[data_names[0]]['file_path']
        aux_data_tr_file = data_stat[data_names[1]]['file_path']

        tar_key_words = data_stat[data_names[0]]['file_key_words']
        aux_key_words = data_stat[data_names[1]]['file_key_words']

        tar_stat = analyze_csv(tar_data_tr_file, tar_key_words)
        aux_stat = analyze_csv(aux_data_tr_file, aux_key_words)

        for key_word in tar_stat.keys():
            for sub_key_word in tar_stat[key_word]:
                if sub_key_word != "info":
                    print(f'tar has {key_word}_{sub_key_word} as, ', tar_stat[key_word][sub_key_word])

        for key_word in aux_stat.keys():
            for sub_key_word in aux_stat[key_word]:
                if sub_key_word != "info":
                    print(f'aux has {key_word}_{sub_key_word} as, ', aux_stat[key_word][sub_key_word])

        ratio = list(range(0, 128, 1))
        tar_sam_num = tar_stat[tar_key_words[-1]]['info'].__len__()
        aux_sam_num = aux_stat[aux_key_words[-1]]['info'].__len__()

        # ratio = [16]
        ratio_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []

        for sub_ratio in ratio:
            filter_tar_sam_num = len(np.where(tar_stat[tar_key_words[-1]]['info'] < sub_ratio)[0])
            filter_aux_sam_num = len(np.where(aux_stat[aux_key_words[-1]]['info'] < sub_ratio)[0])
            if (filter_tar_sam_num + filter_aux_sam_num) == 0 or tar_sam_num == 0:
                continue
            precision = filter_tar_sam_num / (filter_tar_sam_num + filter_aux_sam_num)
            recall = filter_tar_sam_num / (tar_sam_num)
            f1_score = 2 * precision * recall / (precision + recall)
            print(f'in sub_ratio {sub_ratio}, precision is {precision}, recall is {recall}, f1 is {f1_score}')
            ratio_list.append(sub_ratio)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
        print(f'total sample is, {filter_tar_sam_num+filter_aux_sam_num}')
        # in sub_ratio 15, precision is 0.9905624976546963, recall is 0.9786638490342194, f1 is 0.9845772257655441
        # in sub_ratio 16, precision is 0.9900547547211979, recall is 0.9854298743187632, f1 is 0.9877369007803789

        plt.subplot(row_num, col_num, jsq + 1)
        colorlist = ['b', 'r', 'y']
        labellist = ['precision', 'recall', 'F1']
        line_style_list = ['-', '-.', ':']
        x = ratio_list
        y = [precision_list, recall_list, f1_score_list]
        for i in range(len(y)):
            plt.plot(x, y[i], color=colorlist[i], linestyle=line_style_list[i], label=labellist[i])
        plt.xlabel('Pr/Re/F1')
        plt.xlabel('text_len_upper_limitation')
        plt.legend(loc=4, fontsize=8)
        plt.legend(loc='lower left')
        plt.grid(color="k", linestyle="-")
        plt.title(f"{data_names[0]}_{data_names[1]}")
        jsq += 1

img_folder = './img_res'
img_name = "text_len_F1.pdf"
res.savefig(os.path.join(img_folder, img_name), bbox_inches='tight')
print('save finished!')


