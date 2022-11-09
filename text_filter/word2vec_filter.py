import os
import numpy as np

import sys
import torch

sys.path.append('..')
import select_aux_sample
import prepare_aux_text_embedding



def word2vec_sim_filter(hparams, train_data, merge_data, run_opts=None):
    # hparams: parameters in the main file
    # train_data: should set as T2 data, and can only use its text info
    # merge_data: T1 + A data, and can only use its text inf0

    ## 3.1 if the pre-calculated representations do not exist,
    ## calculate text representation for texts in both target dataset and auxiliary dataset
    print("start cal and save pre-calculated word2vec representations for train_data and aux_train_data")
    hparams['word2vec_fea_file'] = os.path.join(hparams['output_folder'], hparams['word2vec_name']+'train.json')
    hparams['aux_word2vec_fea_file'] = os.path.join(hparams['filter_res_by_text_similary_dir'], hparams['word2vec_name'] + 'train.json')
    print('word2vec file is, ', hparams['word2vec_fea_file'])
    print('aux_word2vec file is, ', hparams['aux_word2vec_fea_file'])
    if 'gaulabel' in hparams['text_filter_type']: #or 'random' in hparams['text_filter_type']:
        hparams['semantic2vec_fea_file'] = os.path.join(hparams['output_folder'],
                                                        hparams['word2vec_name'] + 'semantic_fea.json')
        print('semantic2vec file is, ', hparams['semantic2vec_fea_file'])
    if 'divaud' in hparams['text_filter_type']: #or 'random' in hparams['text_filter_type']:
        hparams['aud2vec_fea_file'] = os.path.join(hparams['output_folder'],
                                                        hparams['word2vec_name'] + 'aud_low_fea.json')
        print('aud2vec file is, ', hparams['aud2vec_fea_file'])
        hparams['aux_aud2vec_fea_file'] = os.path.join(hparams['filter_res_by_text_similary_dir'],
                                                        hparams['word2vec_name'] + 'aud_low_fea.json')
        print('aux_aud2vec file is, ', hparams['aux_aud2vec_fea_file'])
        pretrained_hubert = prepare_aux_text_embedding.load_hubert(hparams, run_opts) # load hubert

    ## whether recalculate a word2vec
    if hparams['recalculate_word2vec'] or hparams['aux_recalculate_word2vec']:
        # load a general pre-trained embeddings
        print("start preparing pre-trained word2vec weight file")
        pretrained_word2vec = prepare_aux_text_embedding.prepare_word2vec(hparams)
        if hparams['recalculate_word2vec']:
            # cal audio fea
            if 'divaud' in hparams['text_filter_type']:
                prepare_aux_text_embedding.delete_file(hparams['aud2vec_fea_file'])
                print('start cal audio features of target data')
                prepare_aux_text_embedding.cal_aud2vec_on_csv(hparams, hparams['data_name'], train_data, \
                                                               hparams['aud2vec_fea_file'], pretrained_hubert)

            # cal semantics fea & statistics
            if 'gaulabel' in hparams['text_filter_type']:
                prepare_aux_text_embedding.delete_file(hparams['semantic2vec_fea_file'])
                print('start cal semantic entities features of target data')
                prepare_aux_text_embedding.cal_semantic2vec_on_csv_common_cov(hparams, hparams['data_name'], train_data, \
                                                               hparams['semantic2vec_fea_file'], pretrained_word2vec)

            # cal text fea
            prepare_aux_text_embedding.delete_file(hparams['word2vec_fea_file'])
            print('start cal transcript features of target data')
            prepare_aux_text_embedding.cal_word2vec_on_csv(hparams, hparams['data_name'], train_data, \
                                                           hparams['word2vec_fea_file'], pretrained_word2vec)

        if hparams['aux_recalculate_word2vec']:
            if 'divaud' in hparams['text_filter_type']:
                # pretrained_hubert = prepare_aux_text_embedding.prepare_hubert(hparams)
                prepare_aux_text_embedding.delete_file(hparams['aux_aud2vec_fea_file'])
                print('start cal audio features of aux data')
                prepare_aux_text_embedding.cal_aud2vec_on_csv(hparams, 'merge', merge_data, \
                                                               hparams['aux_aud2vec_fea_file'], pretrained_hubert)

            prepare_aux_text_embedding.delete_file(hparams['aux_word2vec_fea_file'])
            print('start cal transcript features of aux data')
            prepare_aux_text_embedding.cal_word2vec_on_csv(hparams, 'merge', merge_data, \
                                                           hparams['aux_word2vec_fea_file'], pretrained_word2vec)
        # del audio pretrained model and save GPU memory
        if 'divaud' in hparams['text_filter_type']:
            del pretrained_hubert
        torch.cuda.empty_cache()

    else:
        if not os.path.exists(hparams['word2vec_fea_file']) or not os.path.exists(hparams['aux_word2vec_fea_file']):
            raise ValueError(hparams['word2vec_fea_file'] +" or " + hparams['aux_word2vec_fea_file'] + ' not exist!')
        else:
            print(f"{hparams['word2vec_fea_file']} alerady exists")
            print(f"{hparams['aux_word2vec_fea_file']} alerady exists")
        if 'gaulabel' in hparams['text_filter_type']:
            if not os.path.exists(hparams['semantic2vec_fea_file']):
                raise ValueError(hparams['semantic2vec_fea_file'] + ' not exist!')
            else:
                print(f"{hparams['semantic2vec_fea_file']} alerady exists")
        if 'divaud' in hparams['text_filter_type']:
            if not os.path.exists(hparams['aux_aud2vec_fea_file']) or not os.path.exists(hparams['aud2vec_fea_file']):
                raise ValueError(hparams['aux_aud2vec_fea_file'] +" or " + hparams['aud2vec_fea_file'] + ' not exist!')
            else:
                print(f"{hparams['aud2vec_fea_file']} alerady exists")
                print(f"{hparams['aux_aud2vec_fea_file']} alerady exists")




    ## below delect
    # (
    #     aud_sample_label_array_cluster,
    #     # sample_rel_score_array_cluster,
    #     # sample_best_index,
    #     aud_each_cluster_index,
    #     aud_sample_dataset_ids,
    #     aud_cluster_mean_array
    # ) = select_aux_sample.cal_cluster(
    #     hparams,
    #     input_file_name=hparams['aud2vec_fea_file'],
    #     cluster_para=hparams['word2vec_cluster_model'],
    #     fea_keyword='sig_mean_fea_vec',
    #     id_keyword='id',
    #     cluster_save_path=hparams['aud_cluster_res_path']
    # )
    ## abov delect


    ## 3.2.1 calculate text similarity
    # calculate cluster center representation in vs_train_data
    if hparams['recalculate_cluster_mean']:
        (
        sample_label_array_cluster,
        # sample_rel_score_array_cluster,
        # sample_best_index,
        each_cluster_index,
        sample_dataset_ids,
        cluster_mean_array
        ) = select_aux_sample.cal_cluster(
            hparams,
            input_file_name=hparams['word2vec_fea_file'],
            cluster_para=hparams['word2vec_cluster_model'],
            fea_keyword=hparams["word2vec_name"]+'_mean_fea_vec',
            id_keyword='id',
            cluster_save_path=hparams['cluster_res_path']
        )
        cluster_num = cluster_mean_array.shape[0]
        print(f'text cluster num is {cluster_num}')

        if 'divaud' in hparams['text_filter_type']:
            (
                aud_sample_label_array_cluster,
                # sample_rel_score_array_cluster,
                # sample_best_index,
                aud_each_cluster_index,
                aud_sample_dataset_ids,
                aud_cluster_mean_array
            ) = select_aux_sample.cal_cluster(
                hparams,
                input_file_name=hparams['aud2vec_fea_file'],
                cluster_para=hparams['word2vec_cluster_model'],
                fea_keyword='sig_mean_fea_vec',
                id_keyword='id',
                cluster_save_path=hparams['aud_cluster_res_path']
            )
            aud_cluster_num = aud_cluster_mean_array.shape[0]
            print(f'audio cluster num is {aud_cluster_num}')

    else:
        cluster_res = np.load(hparams['cluster_res_path'], allow_pickle=True)
        (
        sample_label_array_cluster,
        # sample_rel_score_array_cluster,
        # sample_best_index,
        each_cluster_index,
        sample_dataset_ids,   # this is the ids index for the samples in the target dataset
        cluster_mean_array
        ) = cluster_res[0], cluster_res[1], cluster_res[2], cluster_res[3]
        cluster_num = cluster_mean_array.shape[0]
        print(f'text cluster num is {cluster_num}')

        if 'divaud' in hparams['text_filter_type']:
            aud_cluster_res = np.load(hparams['aud_cluster_res_path'], allow_pickle=True)
            (
                aud_sample_label_array_cluster,
                # sample_rel_score_array_cluster,
                # sample_best_index,
                aud_each_cluster_index,
                aud_sample_dataset_ids,  # this is the ids index for the samples in the target dataset
                aud_cluster_mean_array
            ) = aud_cluster_res[0], aud_cluster_res[1], aud_cluster_res[2], aud_cluster_res[3]
            aud_cluster_num = aud_cluster_mean_array.shape[0]
            print(f'audio cluster num is {aud_cluster_num}')



    ## 3.2.2 filter by the text similarity
    if hparams["cal_filter_res_by_text_similary"]:
        # the ids in the ini_filter_aux_sample_ids is the respective ids in the aux dataset
        if 'divaud' in hparams['text_filter_type']:
            ini_filter_aux_sample_ids = select_aux_sample.filter_aux_samples(hparams, cluster_mean_array, aud_cluster_mean_array=aud_cluster_mean_array,
                                                                             mode="default")
        else:
            ini_filter_aux_sample_ids = select_aux_sample.filter_aux_samples(hparams, cluster_mean_array, mode="default")
        # add semantics_filter at below

    else:

        ini_filter_aux_sample_ids = np.load(hparams['ini_text_filter_res'], allow_pickle=True).item()
    # print('the nums of aux data to each cluster sentence are below')
    # print([len(ini_filter_aux_sample_ids[str(x)]['ids']) for x in range(cluster_num)])
    print('target_dataset, ', hparams['data_name'])
    print('aux_dataset, ', hparams['aux_data_name'])
    print('cosine ratio, ', hparams['filter_by_sim']['threshold'])
    print('the total rest data is: ', np.array([len(ini_filter_aux_sample_ids[str(x)]['ids']) for x in range(cluster_num)]).sum())
    return ini_filter_aux_sample_ids, cluster_num
