import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os


class DrawMultiViews(object):
    def __init__(self, num_sample_cluster, num_semantic_cat, diagram_save_folder, method_key_word, num_audio_cluster):
        self.num_sample_cluster = num_sample_cluster
        self.num_semantic_cat = num_semantic_cat
        self.diagram_save_folder = diagram_save_folder
        self.method_key_word = method_key_word
        self.num_audio_cluster = num_audio_cluster



    def extract_gt_y(self, gt_data_array, view_key_word):
        if view_key_word == 'num_sample_T2_text_cluster'or view_key_word == 'num_semantics_cluster' or view_key_word=='num_sample_aux_audio_cluster':
            if view_key_word == 'num_sample_T2_text_cluster':
                res = np.zeros(self.num_sample_cluster)
            elif view_key_word == 'num_semantics_cluster':
                res = np.zeros(self.num_semantic_cat)
            elif view_key_word=='num_sample_aux_audio_cluster':
                res = np.zeros(self.num_audio_cluster)
            for ele in list(gt_data_array):
                res[ele] += 1
            res_ratio = res * 1.0 / res.sum()
            return list(res), list(res_ratio)


    def extract_pr_y(self, datadict, view_key_word):
        if view_key_word == 'num_sample_T2_text_cluster' or view_key_word == 'num_semantics_cluster' or view_key_word=='num_sample_aux_audio_cluster':
            if view_key_word == 'num_sample_T2_text_cluster':
                res = np.zeros(self.num_sample_cluster)
            elif view_key_word == 'num_semantics_cluster':
                res = np.zeros(self.num_semantic_cat)
            elif view_key_word=='num_sample_aux_audio_cluster':
                res = np.zeros(self.num_audio_cluster)
            for s_key, s_val in datadict.items():
                res[s_val] += 1
            res_ratio = res * 1.0 / res.sum()
            return list(res * -1), list(res_ratio * -1)

    def get_save_file_name(self, view_key_word, save_name):

        key_word = view_key_word+'.pdf'
        if save_name == None:
            ini_file_name = self.method_key_word+ '_' + key_word
        else:
            ini_file_name = self.method_key_word + '_' + save_name + '_' + key_word

        final_file_name= os.path.join(self.diagram_save_folder, ini_file_name)
        return final_file_name

    def cal_entropy_socre(self, score_list):
        pre_score_array = np.absolute(np.array(score_list))
        pre_score_list = list(pre_score_array)
        final_res = 0
        for x in pre_score_list:
            if x > 0:
                final_res += (-1.0 * x * np.log(x))
        return final_res

    def cal_diver_score(self, score_list):
        # score_list should be a percentage
        pre_score_array = np.absolute(np.array(score_list))
        ref_score_array = np.ones_like(pre_score_array) * (1.0 / len(score_list))
        ini_res = ref_score_array - pre_score_array
        final_res = np.absolute(ini_res)
        return np.sum(final_res)



    def draw_overlapped_bar(self, gt_data_array, pr_data_dict, view_key_word, save_name=None):
        if view_key_word == 'num_sample_T2_text_cluster':
            x = list(range(0, self.num_sample_cluster))
            label_gt = '#samples of T2 based on T2 text clusters'
            label_pr = '#samples of SF text based on T2 text clusters'
        elif view_key_word == 'num_semantics_cluster':
            x = list(range(0, self.num_semantic_cat))
            label_gt = '#samples of T2 based on T2 semantics clusters'
            label_pr = '#samples of SF text based on T2 semantics clusters'
        elif view_key_word == 'num_sample_aux_audio_cluster':
            x = list(range(0, self.num_audio_cluster))
            label_gt = '#samples of Ori Aux based on Ori Aux audio clusters'
            label_pr = '#samples of SF Aux based on Ori Aux audio clusters'
        y_gt, y_gt_ratio = self.extract_gt_y(gt_data_array, view_key_word)
        # print(pr_data_dict)
        # print(view_key_word)
        y_pr, y_pr_ratio = self.extract_pr_y(pr_data_dict, view_key_word)

        # gt_div_score = self.cal_diver_score(y_gt_ratio)
        # pr_div_score = self.cal_diver_score(y_pr_ratio)
        # plt.bar(x, y_gt_ratio, label=label_gt+f" div_score: {np.around(gt_div_score, 3)}", fc='b')
        # plt.bar(x, y_pr_ratio, label=label_pr+f" div_socre: {np.around(pr_div_score, 3)}", fc='r')

        gt_div_score = self.cal_entropy_socre(y_gt_ratio)
        pr_div_score = self.cal_entropy_socre(y_pr_ratio)
        plt.bar(x, y_gt_ratio, label=label_gt+f" entropy: {np.around(gt_div_score, 3)}", fc='b')
        plt.bar(x, y_pr_ratio, label=label_pr+f" entropy: {np.around(pr_div_score, 3)}", fc='r')

        if save_name == None:
            plt.title(view_key_word)
        else:
            plt.title(view_key_word + "_" + save_name)
        plt.legend()

        save_file =self.get_save_file_name(view_key_word, save_name)
        plt.savefig(save_file, bbox_inches='tight')

        plt.show()

        print(f'fig finsihed of {view_key_word}, saved to, {save_file}')
        print(f'{view_key_word}_{save_name} has gt_div_score and pr_div_score as {gt_div_score} and {pr_div_score}.')
        return gt_div_score, pr_div_score



