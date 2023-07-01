import json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance
import os

def cal_cluster(hparams, input_file_name, cluster_para, fea_keyword, id_keyword, cluster_save_path):

    if cluster_para['name'] == 'dbscan':
        cluster_model = DBSCAN(eps=cluster_para['eps'], min_samples=cluster_para['min_samples'], metric=cluster_para['metric'])
    elif cluster_para['name'] == 'kmeans':
        cluster_model = KMeans(n_clusters=cluster_para['n_clusters'], n_init=cluster_para['n_clusters'])
    else:
        print("The cluster_para name is wrong!")
        raise ValueError

    # read json file
    sample_sentence_lists = []
    sample_dataset_ids = []
    with open(input_file_name, encoding='utf-8') as f:
        sample_lines = f.readlines()
        for sample_line in sample_lines:
            sample = json.loads(sample_line)
            sample_dataset_ids.append(sample[id_keyword])
            sample_sentence_lists.append(sample[fea_keyword])
            if True in np.isnan(np.array(sample[fea_keyword])):
                print("there is nan in sample_sentence_lists")
                raise ValueError
    sample_sentence_arrays = np.array(sample_sentence_lists)

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


    each_cluster_index = []
    cluster_mean_list = []
    for i in label_list:
        mid_loc = np.where(label_array==i)
        mid_loc = mid_loc[0]
        mid_vectors = sample_sentence_arrays[mid_loc]
        mid_cluster_mean = mid_vectors.mean(axis=0)
        cluster_mean_list.append(mid_cluster_mean)

        mid_cluster_index = mid_loc

        each_cluster_index.append(mid_cluster_index)
        # rel_score[mid_loc] = distance.cosine(mid_cluster_mean, sample_sentence_arrays[mid_loc])
    cluster_mean_array = np.array(cluster_mean_list)
    cluster_res = []
    cluster_res.append(label_array)
    cluster_res.append(each_cluster_index)
    cluster_res.append(sample_dataset_ids)
    cluster_res.append(cluster_mean_array)

    np.save(cluster_save_path, cluster_res)
    print(f'save cluster_res successfully to ', cluster_save_path)
    return (
            label_array,
            # rel_score,
            # represent_vec_best,
            each_cluster_index,
            sample_dataset_ids,
            cluster_mean_array
            )


def cal_ele_ratio(input_list):
    len_list = [len(ele) for ele in input_list]
    len_array = np.array(len_list)
    sum_len_array = len_array.sum()
    ratio_list = list(len_array * 1.0 / sum_len_array)
    return ratio_list



def filter_aux_samples(hparams, cluster_mean_array, aud_cluster_mean_array=None, mode='default'):
    if mode == 'default':
        input_file_name = hparams['aux_word2vec_fea_file']
    elif mode == 'debug':
        input_file_name = hparams['word2vec_fea_file']
    else:
        print("the mode in filter_aux_samples is wrong")
        raise ValueError

    # read json file & filter by text similarity
    num_cluster = cluster_mean_array.shape[0]
    ini_filter_res = {}
    ini_filter_res['transcript_fea'] = {}

    ini_filter_res['text2transcript_dis'] = {}
    ini_filter_res['text2transcript_cluster_id'] = {}

    # read audio data
    if 'divaud' in hparams['text_filter_type']:

        with open(hparams['aux_aud2vec_fea_file'], encoding='utf-8') as aud_f:
            aud_sample_lines = aud_f.readlines()

        ini_filter_res['aud2wav_dis'] = {}
        ini_filter_res['aud2wav_cluster_id'] = {}
        ini_filter_res['audio_fea'] = {}

    if 'gaulabel' in hparams['text_filter_type']:
        ini_filter_res['text2sem_dis'] = {}
        ini_filter_res['text2sem_cluster_id'] = {}

    for i in range(num_cluster):
        ini_filter_res[str(i)] = {}
        ini_filter_res[str(i)]['ids'] = []
        ini_filter_res[str(i)]['scores'] = []


    if 'gaulabel' in hparams['text_filter_type']:
        # check the statistics of semantic2vec_fea_file
        invalid_cov_mat = []
        with open(hparams['semantic2vec_fea_file'], encoding='utf-8') as f:
            semantic_info_lines = f.readlines()
            assert len(semantic_info_lines) == 1
            for semantic_info in semantic_info_lines:
                semantic_info = json.loads(semantic_info)
                for sub_key, sub_value in semantic_info.items():
                    if 'fea_vec' in sub_key or 'mat_inv' in sub_key:
                        mid_array = np.array(sub_value)
                        if np.isnan(mid_array).any():
                            invalid_cov_mat.append(sub_key)
                sem_type_list = semantic_info['semantic_keywords']

    # check whether wrongly set the text_filter_type
    # assert not ('random' in hparams['text_filter_type'] and 'word2vec' in hparams['text_filter_type'])

    with open(input_file_name, encoding='utf-8') as f:
        sample_lines = f.readlines()
        total_aux_samples = len(sample_lines)
        jsq = -1
        for sample_line in sample_lines:
            jsq += 1
            if mode == 'debug' or hparams['small_set']==True:
                if jsq % 1000 == 0:
                    print(f'finish filter by text similarity of {jsq}/{total_aux_samples}')
                if jsq == 5000:
                    break
            if jsq % 10000 == 0:
                print(f'finish filter by text similarity of {jsq}/{total_aux_samples}')
            chosen_flag = False
            if 'random' in hparams['text_filter_type']:
                random_val = np.random.uniform()
                if random_val < hparams['text_filter_random_ratio']:
                    # print(random_val)
                    chosen_flag = True
                else:
                    continue

            sample = json.loads(sample_line)
            text2transcript_dis_list = []

            for i in range(num_cluster):
                cluster_mean = cluster_mean_array[i]
                mid_distance = distance.cosine(cluster_mean, sample[hparams["word2vec_name"] + '_mean_fea_vec'])
                text2transcript_dis_list.append(mid_distance)
                if chosen_flag == False:
                    if 'word2vec' in hparams['text_filter_type'] and mid_distance <= hparams['filter_by_sim']['threshold'] and 'random' not in hparams['text_filter_type']:
                        chosen_flag = True
                        continue
                    # # below is further filter mid-results in the process
                    # if 'gaulabel' in hparams['text_filter_type']:
                    #     text2semantics_dis_dict = cal_text2semantics_dis_dict(hparams, sample[hparams["word2vec_name"] + '_mean_fea_vec'], semantic_info, sem_type_list, invalid_cov_mat)
                    #     ini_filter_res['text2sem_dis'][str(sample['id'])] = text2semantics_dis_dict
                    #     ini_filter_res['text2sem_cluster_id'][str(sample['id'])] = np.argmin(np.array(text2semantics_dis_dict))
                    # break
            if chosen_flag == True:
                ini_filter_res[str(i)]['ids'].append(sample['id'])
                ini_filter_res[str(i)]['scores'].append(mid_distance)
                ini_filter_res['transcript_fea'][str(sample['id'])] = sample[hparams["word2vec_name"] + '_mean_fea_vec']

                ini_filter_res['text2transcript_dis'][str(sample['id'])] = text2transcript_dis_list
                ini_filter_res['text2transcript_cluster_id'][str(sample['id'])] = np.argmin(np.array(text2transcript_dis_list))

                # cal staticts related to audio
                if 'divaud' in hparams['text_filter_type']:
                    cur_aud_sample = json.loads(aud_sample_lines[jsq])
                    assert cur_aud_sample['id'] == sample['id']
                    ini_filter_res['audio_fea'][str(sample['id'])] = cur_aud_sample['sig' + '_mean_fea_vec']
                    audio2wav_dis_list = [] # audio sample to a audio cluster distance
                    aud_num_cluster = aud_cluster_mean_array.shape[0]
                    for j in range(aud_num_cluster):
                        aud_cluster_mean = aud_cluster_mean_array[j]
                        mid_aud_distance = distance.cosine(aud_cluster_mean, cur_aud_sample['sig' + '_mean_fea_vec'])
                        audio2wav_dis_list.append(mid_aud_distance)
                    ini_filter_res['aud2wav_dis'][str(sample['id'])] = audio2wav_dis_list
                    ini_filter_res['aud2wav_cluster_id'][str(sample['id'])] = np.argmin(np.array(audio2wav_dis_list))

                # cal staticts related to semantics
                if 'gaulabel' in hparams['text_filter_type']:
                    text2semantics_dis_list = cal_text2semantics_dis_dict(hparams, sample[hparams["word2vec_name"] + '_mean_fea_vec'], semantic_info, sem_type_list, invalid_cov_mat)
                    ini_filter_res['text2sem_dis'][str(sample['id'])] = text2semantics_dis_list
                    ini_filter_res['text2sem_cluster_id'][str(sample['id'])] = np.argmin(np.array(text2semantics_dis_list))

    if mode == "default":
        if not os.path.isdir(hparams['filter_res_by_text_similary_dir']):
            os.makedirs(hparams['filter_res_by_text_similary_dir'])
        np.save(hparams['ini_text_filter_res'], ini_filter_res)
    print("initial_filter_res has been finished")
    return ini_filter_res

def cal_M_distance(sample_fea, gau_mean, gau_cov_inv):
    dif = sample_fea - gau_mean
    res = dif.dot(gau_cov_inv)
    res = res.dot(dif.T)
    return res[0, 0]

def cal_eu_distance(sample_fea, gau_mean):
    dif = sample_fea - gau_mean
    res = dif.dot(dif.T)
    return res[0, 0]

def cal_text2semantics_dis_dict(hparams, text_fea, sem_fea_dict, sem_type_list, invalid_cov_list=None):
    # semantic_token_id = sub_type + 'token_id'
    res = []
    text_fea = np.array(text_fea)
    text_fea = np.expand_dims(text_fea, axis=0)

    use_individual_sem_inv = False

    for sub_type in sem_type_list:
        semantic_mean_fea_vec_key = hparams["word2vec_name"] + '_' + sub_type + '_mean_fea_vec'
        if use_individual_sem_inv:
            semantic_cov_mat_inv_key = hparams["word2vec_name"] + '_' + sub_type + '_cov_mat_inv' # inv_mat for individual semantics
        else:
            semantic_comb_cov_mat_inv_key = hparams["word2vec_name"] + '_' + 'combined' + '_cov_mat_inv' # inv_mat for whole semantics

        semantic_mean_fea_vec = sem_fea_dict[semantic_mean_fea_vec_key]
        if use_individual_sem_inv:
            semantic_cov_mat_inv = sem_fea_dict[semantic_cov_mat_inv_key] # inv_mat for individual mat
        else:
            semantic_comb_cov_mat_inv = sem_fea_dict[semantic_comb_cov_mat_inv_key]

        semantic_mean_fea_vec = np.array(semantic_mean_fea_vec)
        semantic_mean_fea_vec = np.expand_dims(semantic_mean_fea_vec, axis=0)
        if not use_individual_sem_inv:
            text_2sin_sem_dis = cal_M_distance(text_fea, semantic_mean_fea_vec, semantic_comb_cov_mat_inv)
        else:
            # for individual semantics
            if semantic_cov_mat_inv_key in invalid_cov_list:
                text_2sin_sem_dis = cal_eu_distance(text_fea, semantic_mean_fea_vec)
            else:
                semantic_cov_mat_inv = np.array(semantic_cov_mat_inv)
                text_2sin_sem_dis = cal_M_distance(text_fea, semantic_mean_fea_vec, semantic_cov_mat_inv)
        res.append(text_2sin_sem_dis)
    return res




def cal_expect_cluster_sample_num(label_ratio, cluster_sample_num):
    label_ratio = np.array(label_ratio)
    cluster_sample_num = np.array(cluster_sample_num)
    cluster_num = len(label_ratio)
    cluster_id = list(range(cluster_num))
    label_ratio_sort_index = np.argsort(label_ratio)
    res = None
    for i in label_ratio_sort_index:
        expect_total_sample_num = int(cluster_sample_num[i]/label_ratio[i])
        expect_each_cluster_sample_num = expect_total_sample_num * label_ratio
        diff = cluster_sample_num - expect_each_cluster_sample_num
        judge = np.where(diff < 0, 1, 0).sum()
        if judge > 0:
            res = expect_each_cluster_sample_num
            return res
    if res == None:
        print('the label ratio has no results')
        return res


def select_filter_res(hparams, ini_filter_res, cluster_sample_index):
    # filter by label ratio
    num_cluster = cluster_sample_index
    label_ratio = cal_ele_ratio(cluster_sample_index)
    # calculate the expected sample numbers from the highest ratio to the lowest ratio.
    cluster_sample_num = [len(ini_filter_res[str(i)]['ids']) for i in range(num_cluster)]
    expect_cluster_files_distribution = cal_expect_cluster_sample_num(label_ratio, cluster_sample_num)
    if expect_cluster_files_distribution == None:
        refined_filter_res = ini_filter_res
        return refined_filter_res
    else:
        refined_filter_res = {}
        for i in range(num_cluster):
            refined_filter_res[str(i)] = {}
            # cal ranked scores from small number to big number
            mid_cluser_sample_rank = np.args(ini_filter_res[str(i)]['scores'])
            # only extract the first part of "scores".
            mid_scores = []
            mid_ids = []
            for ele in mid_cluser_sample_rank[0:expect_cluster_files_distribution[i]]:
                mid_scores.append(ini_filter_res[str(i)]['scores'][ele])
                mid_ids.append(ini_filter_res[str(i)]['ids'][ele])
            refined_filter_res[str(i)]['scores'] = mid_scores
            refined_filter_res[str(i)]['ids'] = mid_ids
        return refined_filter_res
