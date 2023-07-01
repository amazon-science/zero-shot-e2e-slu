import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
import random

# sys.path.append('..')
# import select_aux_sample
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance


class MultiViewSampleFilter(object):
    def __init__(self, num_text_cluster, num_semantic_cat, num_audio_cluster,
                 text_fea_clusters, aud_fea_clusters, semantic_fea_cls_centers,
                 ini_res_dict, text2text_clusters_key, aud2aud_clusters_key, text2semantic_cls_centers_key,
                 cluster_para, balance_save_npy, load_precal_multi_view_clusters, mulview_sf_weights):
        self.num_text_cluster = num_text_cluster  # for text sample
        self.num_semantic_cat = num_semantic_cat
        self.num_audio_cluster = num_audio_cluster
        # load cluster_rep(3view) & filter_fea
        self.text_fea_clusters = text_fea_clusters
        self.aud_fea_clusters = aud_fea_clusters
        self.semantic_fea_cls_centers = semantic_fea_cls_centers
        # load filter_fea
        self.ini_res_dict = ini_res_dict
        self.text2text_clusters_key = text2text_clusters_key
        self.aud2aud_clusters_key = aud2aud_clusters_key
        self.text2semantic_cls_centers_key = text2semantic_cls_centers_key
        self.cluster_para = cluster_para
        self.balance_save_npy = balance_save_npy
        self.load_precal_multi_view_clusters = load_precal_multi_view_clusters
        self.mulview_sf_weights = mulview_sf_weights

        self.text2text_clusters = self.ini_res_dict[self.text2text_clusters_key]
        self.aud2aud_clusters = self.ini_res_dict[self.aud2aud_clusters_key]
        self.text2semantic_cls_centers = self.ini_res_dict[self.text2semantic_cls_centers_key]

        self.id_str_list = list(self.text2text_clusters.keys())
        self.check_data(self.id_str_list, list(self.aud2aud_clusters.keys()))
        self.check_data(self.id_str_list, list(self.text2semantic_cls_centers.keys()))



        self.clus_id2sample_id = {}
        for i in range(len(self.id_str_list)):
            self.clus_id2sample_id[str(i)] = self.id_str_list[i]

    def check_data(self, id_str_list1, id_str_list2):
        assert len(id_str_list1) == len(id_str_list2)

        id_int_list1 = [int(ele) for ele in id_str_list1]
        id_int_list2 = [int(ele) for ele in id_str_list2]
        id_int_arr1 = np.array(id_int_list1).sort()
        id_int_arr2 = np.array(id_int_list2).sort()
        assert id_int_arr1 == id_int_arr2

    def dict2fea(self, id_str_list, fea_dict):
        res = []
        for id_str in id_str_list:
            mid_fea = fea_dict[id_str]
            res.append(mid_fea)
        res = np.array(res)
        print("extracted_dist_array_shape: ", res.shape)
        return res

    def standard_array(self, array_2d):
        def gau_norm(dt):
            mu = np.mean(dt)
            sigma = np.std(dt)
            return (dt-mu) * 1.0 /sigma
        res = gau_norm(array_2d)
        print("standard_dist_array_shape: ", res.shape)
        return res

    def build_multi_view_embed(self):
        # build text2transcript view
        ini_text2transcript_emb = self.dict2fea(self.id_str_list, self.text2text_clusters)
        std_text2transcript_emb = self.standard_array(ini_text2transcript_emb) * self.mulview_sf_weights['transcript']
        # build aud2aud view
        ini_aud2aud_emb = self.dict2fea(self.id_str_list, self.aud2aud_clusters)
        std_aud2aud_emb = self.standard_array(ini_aud2aud_emb) * self.mulview_sf_weights['audio']
        # build text2semantic view
        ini_text2semantic_emb = self.dict2fea(self.id_str_list, self.text2semantic_cls_centers)
        std_text2semantic_emb = self.standard_array(ini_text2semantic_emb) * self.mulview_sf_weights['semantic']

        self.std_multi_view_emb = np.concatenate((std_text2transcript_emb, std_aud2aud_emb, std_text2semantic_emb), axis=1)
        print('multi_view_embedding has shape of, ', self.std_multi_view_emb.shape)

        return self.std_multi_view_emb

    def cluster_multi_view_embed(self):
        if self.load_precal_multi_view_clusters == False:
            sample_dataset_ids = self.id_str_list
            if self.cluster_para['name'] == 'dbscan':
                cluster_model = DBSCAN(eps=self.cluster_para['eps'], min_samples=self.cluster_para['min_samples'],
                                       metric=self.cluster_para['metric'])
            elif self.cluster_para['name'] == 'kmeans':
                cluster_model = KMeans(n_clusters=self.cluster_para['n_clusters'], n_init=self.cluster_para['n_clusters'])
            else:
                print("The cluster_para name is wrong!")
                raise ValueError

            sample_sentence_arrays = self.std_multi_view_emb

            # cluster fit
            fitted_clusters = cluster_model.fit(sample_sentence_arrays)
            label_array = np.array(fitted_clusters.labels_)
            label_list = list(set(fitted_clusters.labels_))
            data_num, data_dim = sample_sentence_arrays.shape
            try:
                invalid_loc = label_list.index(-1)
                label_list.pop(invalid_loc)
            except:
                pass

            # judge whether the cluster results is valid
            cluster_flag = True
            rel_score = represent_vec_best = each_cluster_index = None
            if -1 not in label_list and len(label_list) == 0:
                cluster_flag = False
            elif -1 in label_list and len(label_list) == 1:
                cluster_flag = False
            if cluster_flag == False:
                print('The cluster is failed, please reset the cluster parameters!')
                raise ValueError

            ## process cluster results
            each_cluster_index = []
            cluster_mean_list = []

            for i in label_list:
                mid_loc = np.where(label_array == i)
                mid_loc = mid_loc[0]
                mid_vectors = sample_sentence_arrays[mid_loc]
                mid_cluster_mean = mid_vectors.mean(axis=0)
                cluster_mean_list.append(mid_cluster_mean)
                mid_cluster_index = mid_loc
                each_cluster_index.append(mid_cluster_index)

            cluster_mean_array = np.array(cluster_mean_list)
            cluster_res = []
            cluster_res.append(label_array)
            cluster_res.append(each_cluster_index)
            cluster_res.append(sample_dataset_ids)
            cluster_res.append(cluster_mean_array)
            cluster_num = cluster_mean_array.shape[0]
            np.save(self.balance_save_npy, cluster_res)
            print(f'save multi_view cluster_res successfully to ', self.balance_save_npy)
        else:
            print('skip cal multi-view cluster res and load a pre-cal ones')
            cluster_res = np.load(self.balance_save_npy, allow_pickle=True)
            (
                label_array,
                each_cluster_index,
                sample_dataset_ids,
                cluster_mean_array
            ) = cluster_res[0], cluster_res[1], cluster_res[2], cluster_res[3]
            cluster_num = cluster_mean_array.shape[0]

        print(f'multi-view cluster num is {cluster_num}')

        self.multi_view_label_array = label_array
        self.multi_view_each_cluster_index = each_cluster_index
        self.multi_view_sample_dataset_ids = sample_dataset_ids
        self.multi_view_cluster_mean_array = cluster_mean_array
        return (
            label_array,
            each_cluster_index,
            sample_dataset_ids,
            cluster_mean_array
        )

    def cal_balanced_sample_ids(self, mulview_sf_number=None, mulview_filter_mode='balance'):
        cluster_lengths = []
        multi_view_each_cluster_index = list(self.multi_view_each_cluster_index)
        for ele in multi_view_each_cluster_index:
            cluster_lengths.append(len(ele))

        total_num = np.array(cluster_lengths).sum()
        if mulview_sf_number != None and mulview_sf_number > total_num:
            raise ValueError('mulview_sf_number > total_num, please reset mulview_sf_number')
        min2max_index = np.array(cluster_lengths).argsort()
        min2max_num = copy.deepcopy(np.array(cluster_lengths))
        min2max_num.sort()
        min2max_num = list(min2max_num)
        min2max_index = list(min2max_index)
        min_cluster_num = np.array(cluster_lengths).min()  # the min size requirement for each cluster size
        assert min_cluster_num == min2max_num[0]

        if mulview_filter_mode == 'dynamic_balance':
            assert mulview_sf_number != None
            cluster_number = len(multi_view_each_cluster_index)
            expect_min_cluster_num = int(mulview_sf_number / cluster_number)
            chose_id = []
            pop_list = []
            res_len_list = []
            deduct_cluster_number = copy.deepcopy(cluster_number)
            deduct_mulview_sf_number = copy.deepcopy(mulview_sf_number)
            while min2max_num[0] < expect_min_cluster_num:

                cur_index = min2max_index[0]
                pop_list.append(cur_index)
                res_len_list.append(len(multi_view_each_cluster_index[cur_index]))
                deduct_mulview_sf_number = deduct_mulview_sf_number - len(multi_view_each_cluster_index[cur_index])
                chose_id.extend([int(self.clus_id2sample_id[str(sub_extr_ele)]) for sub_extr_ele in multi_view_each_cluster_index[cur_index]])

                min2max_index.pop(0)
                min2max_num.pop(0)

                deduct_cluster_number = deduct_cluster_number - 1
                expect_min_cluster_num = int(deduct_mulview_sf_number / deduct_cluster_number)

            for i in range(len(multi_view_each_cluster_index)):
                if i not in pop_list:
                    ele = multi_view_each_cluster_index[i]
                    extr_ele = ele[0:expect_min_cluster_num]
                    chose_id.extend([int(self.clus_id2sample_id[str(sub_extr_ele)]) for sub_extr_ele in extr_ele])
                    res_len_list.append(expect_min_cluster_num)
            print(f'dynamic_balance has the cluster pick result in length as {res_len_list}')
            assert abs(np.array(res_len_list).sum() - mulview_sf_number) <= cluster_number



        elif mulview_filter_mode == 'balance':

            cluster_number = len(multi_view_each_cluster_index)
            expect_min_cluster_num = int(mulview_sf_number/cluster_number)
            if mulview_sf_number!= None and min_cluster_num > expect_min_cluster_num:
                min_cluster_num = expect_min_cluster_num
            elif mulview_sf_number== None:
                pass
            else:
                print(f'the min_cluster_num is {min_cluster_num}')
                raise ValueError(f'the mulview_sf_number is larger than {min_cluster_num}, please use dynamic balance')
            # extract and project relative ids to the absolute ids
            chose_id = []
            for ele in multi_view_each_cluster_index:
                extr_ele = ele[0:min_cluster_num]
                chose_id.extend([int(self.clus_id2sample_id[str(sub_extr_ele)]) for sub_extr_ele in extr_ele])

        elif mulview_filter_mode == 'extreme':
            chose_id = []
            miss_sample_num = copy.deepcopy(mulview_sf_number)
            if mulview_sf_number != None:
                for ele in multi_view_each_cluster_index[::-1]:
                    # set_max_number = 1200
                    # if len(ele) > set_max_number:
                    #     ele = ele[0:set_max_number]
                    if len(ele) >= miss_sample_num:
                        extr_ele = ele[0:miss_sample_num]
                        miss_sample_num = miss_sample_num - len(extr_ele)
                    else:
                        extr_ele = ele[:]
                        miss_sample_num = miss_sample_num - len(extr_ele)
                    chose_id.extend([int(self.clus_id2sample_id[str(sub_extr_ele)]) for sub_extr_ele in extr_ele])
                    if miss_sample_num <= 0:
                        break
            else:
                raise ValueError('mulview_sf_number cannot be None in extreme mode')

        elif mulview_filter_mode == 'random':
            if mulview_sf_number == None:
                raise ValueError('mulview_sf_number cannot be None in random mode')
            total_chose_id = []
            for ele in multi_view_each_cluster_index:
                total_chose_id.extend(ele)
            mid_id_list = random.sample(total_chose_id, mulview_sf_number)
            chose_id = ([int(self.clus_id2sample_id[str(sub_extr_ele)]) for sub_extr_ele in mid_id_list])


        print(f'use {mulview_filter_mode} mode choose sample in size of {len(chose_id)}')

        # read respective samples' cluster ids

        chose_text2transcript_cluster_id = {}
        chose_adu2wav_cluster_id = {}
        chose_text2sem_cluster_id = {}

        for ele in chose_id:
            chose_text2transcript_cluster_id[str(ele)] = self.ini_res_dict['text2transcript_cluster_id'][str(ele)]
            chose_adu2wav_cluster_id[str(ele)] = self.ini_res_dict['aud2wav_cluster_id'][str(ele)]
            chose_text2sem_cluster_id[str(ele)] = self.ini_res_dict['text2sem_cluster_id'][str(ele)]


        self.chose_id = chose_id
        self.chose_text2transcript_cluster_id = chose_text2transcript_cluster_id
        self.chose_adu2wav_cluster_id = chose_adu2wav_cluster_id
        self.chose_text2sem_cluster_id = chose_text2sem_cluster_id
        return min_cluster_num, chose_id, chose_text2transcript_cluster_id, chose_adu2wav_cluster_id, chose_text2sem_cluster_id






