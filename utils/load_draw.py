import sys

import numpy as np
import os
import json

sys.path.append('..')
from text_filter import draw_plots
from models import multi_view_sample_filter

def load_draw(
    hparams,
    final_filter_aux_sample_dict,
    T2_transcript_cluster_res_path,
    Aux_aud_cluster_res_path,
    T2_semantics_cluster_res_path,
    diagram_save_folder,
    sf_type,
    ):

    # draw diagrams for the SM results
    if 'draw_img' not in hparams:
        hparams['draw_img'] = False
    if hparams['draw_img']:
        # load sample_filter res
        # final_filter_aux_sample_dict = final_filter_aux_sample_ids.item()

        print('start draw diagrams from the multi-views')
        print(final_filter_aux_sample_dict.keys())
        cluster_metrics = ['text_fea', 'sem_dis', 'aud_div']

        # load T2 text cluster results
        cluster_res = np.load(T2_transcript_cluster_res_path, allow_pickle=True)
        (
            sample_label_array_cluster,
            # sample_rel_score_array_cluster,
            # sample_best_index,
            each_cluster_index,
            sample_dataset_ids,  # this is the ids index for the samples in the target dataset
            cluster_mean_array
        ) = cluster_res[0], cluster_res[1], cluster_res[2], cluster_res[3]

        # load T2 audio cluster results
        aud_cluster_res = np.load(Aux_aud_cluster_res_path, allow_pickle=True)
        (
            aud_sample_label_array_cluster,
            # sample_rel_score_array_cluster,
            # sample_best_index,
            aud_each_cluster_index,
            aud_sample_dataset_ids,  # this is the ids index for the samples in the target dataset
            aud_cluster_mean_array
        ) = aud_cluster_res[0], aud_cluster_res[1], aud_cluster_res[2], aud_cluster_res[3]

        # load T2 semantics results
        semantic_cls_fea_list = []
        with open(T2_semantics_cluster_res_path, encoding='utf-8') as f:
            semantic_info_lines = f.readlines()
            assert len(semantic_info_lines) == 1
            for semantic_info in semantic_info_lines:
                semantic_info = json.loads(semantic_info)
                sem_type_list = semantic_info['semantic_keywords']
                sample_semantic_list = semantic_info[
                    'sample_semantic_list']  # will be greater than #sample, because some samples have more than one semantics
                # load semantic class centers
                for sub_key, sub_value in semantic_info.items():
                    if 'fea_vec' in sub_key:
                        semantic_cls_fea_list.append(sub_key)

        # initialize DrawMultiViews class
        # diagram_save_folder = os.path.join(os.getcwd(), hparams['filter_res_by_text_similary_dir'], 'diagram')
        if not os.path.exists(diagram_save_folder):
            os.makedirs(diagram_save_folder)
        dmv = draw_plots.DrawMultiViews(
            num_sample_cluster=cluster_mean_array.shape[0],  # for text sample
            num_semantic_cat=len(sem_type_list),
            diagram_save_folder=diagram_save_folder,
            method_key_word=sys.argv[1][8:-5],
            num_audio_cluster=aud_cluster_mean_array.shape[0],
        )

        # draw bar in terms of #samples in each text cluster
        gt_txt2transcipt_div_score, pr_txt2transcipt_div_score = dmv.draw_overlapped_bar(
            gt_data_array=sample_label_array_cluster,
            pr_data_dict=final_filter_aux_sample_dict['text2transcript_cluster_id'],
            view_key_word='num_sample_T2_text_cluster',
            save_name=sf_type
        )

        # draw bar in terms of #samples in each audio cluster
        gt_adu2wav_div_score, pr_adu2wav_div_score = dmv.draw_overlapped_bar(
            gt_data_array=aud_sample_label_array_cluster,
            pr_data_dict=final_filter_aux_sample_dict['aud2wav_cluster_id'],
            view_key_word='num_sample_aux_audio_cluster',
            save_name=sf_type
        )

        # draw bar in terms of #samples in each semantics
        semantic_dict = {}
        for i in range(len(sem_type_list)):
            semantic_dict[sem_type_list[i]] = i
        processed_sample_semantic_list = [semantic_dict[ele['type']] for ele in sample_semantic_list]
        gt_text2sem_div_score, pr_text2sem_div_score = dmv.draw_overlapped_bar(
            gt_data_array=processed_sample_semantic_list,
            pr_data_dict=final_filter_aux_sample_dict['text2sem_cluster_id'],
            view_key_word='num_semantics_cluster',
            save_name=sf_type
        )

        total_gt_div_score = (gt_txt2transcipt_div_score + gt_adu2wav_div_score + gt_text2sem_div_score) / 3
        total_pr_div_score = (pr_txt2transcipt_div_score + pr_adu2wav_div_score + pr_text2sem_div_score) / 3

        return (total_gt_div_score, total_pr_div_score, cluster_mean_array, sem_type_list, aud_cluster_mean_array,
                sample_label_array_cluster, aud_sample_label_array_cluster, semantic_cls_fea_list, final_filter_aux_sample_dict)


# ----------------------
def multi_view_filter_process(
    hparams,
    cluster_mean_array,
    sem_type_list,
    aud_cluster_mean_array,
    sample_label_array_cluster,
    aud_sample_label_array_cluster,
    semantic_cls_fea_list,
    final_filter_aux_sample_dict,
    mulview_sf_weights,
    mulview_sf_number=None,
    mulview_filter_mode='balance',
    load_precal_multi_view_clusters=False
    ):


        multi_view_sf = multi_view_sample_filter.MultiViewSampleFilter(
            num_text_cluster=cluster_mean_array.shape[0],  # for text sample
            num_semantic_cat=len(sem_type_list),
            num_audio_cluster=aud_cluster_mean_array.shape[0],
            # load cluster_rep(3view) & filter_fea
            text_fea_clusters=sample_label_array_cluster,
            aud_fea_clusters=aud_sample_label_array_cluster,
            semantic_fea_cls_centers=semantic_cls_fea_list,
            # load filter_fea
            ini_res_dict=final_filter_aux_sample_dict,
            text2text_clusters_key='text2transcript_dis',
            aud2aud_clusters_key='aud2wav_dis',
            text2semantic_cls_centers_key='text2sem_dis',
            cluster_para=hparams['multi_view_cluster_model'],
            balance_save_npy=hparams['balance_sf_res_path'],
            load_precal_multi_view_clusters=load_precal_multi_view_clusters,
            mulview_sf_weights=mulview_sf_weights
        )

        # cal multi_view embeddings
        multi_view_emb = multi_view_sf.build_multi_view_embed()

        # cal multi_view cluster_res
        multi_view_cluster_res = multi_view_sf.cluster_multi_view_embed()
        (multi_view_label_array,
         multi_view_each_cluster_index,  # should project to abs sample id based on a list "multi_view_sample_dataset_ids"
         multi_view_sample_dataset_ids,
         multi_view_cluster_mean_array) = multi_view_cluster_res

        # cal balanced sample ids:
        balanced_min_sample_num_in_all_clustets, balanced_selected_sample_dataset_id, chose_text2transcript_cluster_id, chose_adu2wav_cluster_id, chose_text2sem_cluster_id = \
            multi_view_sf.cal_balanced_sample_ids(mulview_sf_number=mulview_sf_number, mulview_filter_mode=mulview_filter_mode)
        return (balanced_min_sample_num_in_all_clustets, balanced_selected_sample_dataset_id, chose_text2transcript_cluster_id, chose_adu2wav_cluster_id, chose_text2sem_cluster_id)

