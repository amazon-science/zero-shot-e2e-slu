######
# This CMSST framework is revised on https://github.com/speechbrain/speechbrain/blob/develop/recipes/SLURP/direct/train_with_wav2vec2.py
######


import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import pandas as pd
from data_process import slurp_process, merge_process, slue_voxpopuli_process
import prepare_process_data
import prepare_aux_text_embedding
import os
import numpy as np
from text_filter import word2vec_filter, text_len_filter
from models import NLU_text2sem, direct_audio2sem, sel_direct_audio2sem
from utils import load_draw


if __name__ == "__main__":



    ### 1. evn_build
    print('------------------------------------------------')
    print('1. evn_build')

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:-1])
    with open(hparams_file) as tar_fin:
        hparams = load_hyperpyyaml(tar_fin, overrides)  # include the processes to load modules

    ### construct ablation study env
    # This is used for copying same data folder (<seed>) from another experiment result fold (if <refer_seed> is not None)
    # If the trained model is not deleted in the copied data folder, below code will prevent the code from continuing
    # You should make sure the trained model ('<save_folder>/direct/') is deleted, if you set <refer_seed> as not None.
    if 'refer_seed' in hparams.keys() and hparams['refer_seed'] != None and hparams['retrain_s3'] == True:
        results_folder = os.path.join(os.getcwd(), 'results')
        prepare_process_data.assign_abstudy_folder(
            main_folder_id=hparams['seed'],
            refer_folder_id=hparams['refer_seed'],
            res_folder_path=results_folder,
            data_name=hparams['data_name'])

    show_results_every = 100  # plots results every N iterations

    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory # several log files are recorded in the "results" folder
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not os.path.exists(hparams['filter_res_by_text_similary_dir']):
        os.mkdir(hparams['filter_res_by_text_similary_dir'])



    ### 2. dataset_build + dataset split
    print('------------------------------------------------')
    print('2. dataset_build + dataset split')
    prepare_process_data.prepare_data(hparams, data_name=hparams['data_name'], data_role='tar', split=hparams['split_ratio'])
    ## 2.1 split Target dataset into two parts (invs_train_data -> S1, vs_train_data -> B1)
    if hparams['operate_divide_csv_train']:
        if hparams['split_ratio']!="None" and hparams["divide_csv_train"]:
            print("start spliting target data by ratio")
            prepare_process_data.split_csv_ratio(hparams["csv_train"], hparams["invs_csv_train"], hparams["vs_csv_train"], hparams["fine_turn_csv"], hparams['split_ratio'])

        elif hparams["divide_csv_train"]:
            print("start spliting target data by non-ratio")
            prepare_process_data.split_csv(hparams["csv_train"], hparams["invs_csv_train"], hparams["vs_csv_train"])
        else:
            raise ValueError('divide_csv_train is not consistent with operate_divide_csv_train')
    else:
        if not (os.path.exists(hparams["invs_csv_train"])
                and os.path.exists(hparams["vs_csv_train"])
                and os.path.exists(hparams["fine_turn_csv"])):
            raise ValueError(
                "divided files are not exist,  and make sure spliting won't change original divided files if set divide_csv_train as True")
        print("skip split target data")

    ## 2.2 read Target dataset
    print("start preparing target data")
    if hparams["divide_csv_train"] and hparams['split_ratio']:
        # vs_train_data -> T2 (D^{t}_{T->M}), invs_train_data -> T1 (D^{t}_{A->T} or D^{t}_{A->S}), fine_tune_data -> T3 (for future usage),
        (vs_train_data, invs_train_data, fine_tune_data, _, test_data, tokenizer) = \
        prepare_process_data.process_data(hparams, data_name=hparams['data_name'], data_role='tar', split_ratio=hparams['split_ratio'])
        train_data = vs_train_data
    elif hparams["divide_csv_train"]:
        # (invs_train_data -> T1 (D^{t}_{A->T} or D^{t}_{A->S}), vs_train_data -> T2 (D^{t}_{T->M}))
        (vs_train_data, invs_train_data, _, test_data, tokenizer) = \
        prepare_process_data.process_data(hparams, data_name=hparams['data_name'], data_role='tar')
        train_data = vs_train_data
    else:
        (train_data, valid_set, test_set, tokenizer) = \
        prepare_process_data.process_data(hparams, data_name=hparams['data_name'], data_role='tar')
    print(f'vs_train_data has length as {len(vs_train_data)}')
    print(f'invs_train_data has length as {len(invs_train_data)}')
    try:
        print(f'fine_tune_data has length as {len(fine_tune_data)}')
    except:
        print('there is no fine_tune_data')

    ## 2.3 read Auxiliary dataset (used for D^{o}_{A->T})
    # if we use MinisPS as Auxiliary dataset in MiniPS2SLURP setting, D^{o}_{A->T} is other domain
    # elif we use VoxPopuli as Auxiliary dataset in VoxPopuli2SLUE setting, D^{o}_{A->T} is the same domain to target domain
    print("start preparing auxiliary data")
    prepare_process_data.prepare_data(hparams, data_name=hparams['aux_data_name'], data_role='aux')
    # 2.4 merge invs_train_data and aux_train_data into a new dataset
       # will remove all semantics from the invs_train_data
    if hparams['aux_data_name'] == 'peoplespeech':
        if hparams['data_name'] == 'slurp':
            refer_slurp_dict = {
                "ID":"ID",
                "duration":"duration",
                "audio_filepath":"wav",
                "text":"transcript"
            }

    elif hparams['aux_data_name'] == 'slue-voxpopuli-full':
        if hparams['data_name'] == 'slue-voxpopuli':
            refer_slurp_dict = {
                "ID": "ID",
                "wav": "wav",
                "transcript":"transcript"
            }
    else:
        raise ValueError('This data combo is not implemented, might be wrong')


    # shuffle merged data with a fixed random seed
    if hparams['merge_tar_aux'] == True:
        if hparams['data_name'] == 'slurp':
            assert hparams['aux_data_name'] == 'peoplespeech'
        elif hparams['data_name'] == 'slue-voxpopuli':
            assert hparams['aux_data_name'] == 'slue-voxpopuli-full'
        else:
            raise ValueError('data_name is wrong/not designed')
        prepare_process_data.merge_tar_aux_data_general(tar_source=hparams['invs_csv_train'], aux_source=hparams['aux_csv_train'], \
                                                save_name=hparams['merge_save_path'], tar_refer_slurp_dict=None, aux_refer_slurp_dict=refer_slurp_dict, tar_name=hparams['data_name'], aux_name=hparams['aux_data_name'])
    else:
        assert os.path.exists(hparams['merge_save_path'])
        print('merge_save_path already exists')

    # check whether the sum of number of invs_csv_train, number of aux_csv_train equals to number of merged data
    invs_train_data_num, aux_train_data_num, mer_train_data_num = \
        prepare_process_data.check_tar_aux_data(tar_source=hparams['invs_csv_train'], aux_source=hparams['aux_csv_train'], save_name=hparams['merge_save_path'])

    # aux_train_data = T1 + A
    merge_data = merge_process.dataio_prepare_text_match(hparams, target_name1=hparams['merge_save_path'], data_role='aux')



    ### 3. process to mine similar audio-texts from Auxiliary dataset
    print('------------------------------------------------')
    print('3. process to mine similar audio-texts from T1+A')
    if 'word2vec' in hparams['text_filter_type'] or 'random' in hparams['text_filter_type']:
        # Sec 4.3: Text similarity based selection
        ini_filter_aux_sample_ids, cluster_num = word2vec_filter.word2vec_sim_filter(hparams, train_data, merge_data, run_opts)
    elif hparams['text_filter_type'] == 'text_len':
        # Use text length for initial selection (not used in the paper)
        ini_filter_aux_sample_ids, cluster_num = text_len_filter.text_len_filter(hparams, train_data, merge_data)

    ## 3.3 load Text similarity based selection results
    final_filter_aux_sample_ids = np.load(hparams['ini_text_filter_res'], allow_pickle=True)

    ## 3.4 build a new subset, part of the auxiliary dataset
    if 'multi_view_balance' not in hparams.keys() or hparams['multi_view_balance'] == False:
        if hparams['recal_filter_csv']:
            selected_sample_dataset_id = []
            for i in range(cluster_num):
                mid_sample_list_id = [int(x) for x in final_filter_aux_sample_ids.item()[str(i)]['ids']]
                if len(mid_sample_list_id) == 0:
                    continue
                mid_sample_dataset_id = [x for x in mid_sample_list_id]
                selected_sample_dataset_id.extend(mid_sample_dataset_id)
            selected_aux_cvs = prepare_process_data.filter_csv(hparams['merge_save_path'], selected_sample_dataset_id, hparams['seleted_aux_csv_wo'])
            print('save filter_csv to: ', hparams['seleted_aux_csv_wo']) # the hparams['seleted_aux_csv_wo'] file has no pseudo semantic labels.
        else:
            print('load filter_csv from: ', hparams['seleted_aux_csv_wo'])

    ### below is the implementation of Multi-view Clustering-based Sample Selection (MCSS) in Sec. 4.3
    # draw text_simlarity filter res distribution
    if 'draw_img' in hparams.keys() and hparams['draw_img']:
        diagram_save_folder = os.path.join(os.getcwd(), hparams['filter_res_by_text_similary_dir'], 'diagram')
        (total_gt_div_score, total_pr_div_score, cluster_mean_array, sem_type_list, aud_cluster_mean_array, sample_label_array_cluster,
        aud_sample_label_array_cluster, semantic_cls_fea_list, final_filter_aux_sample_dict) = load_draw.load_draw(
            hparams,
            final_filter_aux_sample_dict=final_filter_aux_sample_ids.item(),
            T2_transcript_cluster_res_path=hparams['cluster_res_path'],
            Aux_aud_cluster_res_path=hparams['aud_cluster_res_path'],
            T2_semantics_cluster_res_path=hparams['semantic2vec_fea_file'],
            diagram_save_folder=diagram_save_folder,
            sf_type=None # 'balance'
        )
        print(f'basic sf has total gt_div_score and total pr_div_score as {total_gt_div_score} and {total_pr_div_score}.')

        # conduct MCSS
        if 'multi_view_balance' in hparams.keys() and hparams['multi_view_balance'] == True:
            print('------------------------------------------------')
            if 'mulview_filter_mode' not in hparams.keys():
                hparams['mulview_filter_mode'] = 'balance'
            print('Process to balance from three views')
            (balanced_min_sample_num_in_all_clustets, balanced_selected_sample_dataset_id,
             chose_text2transcript_cluster_id, chose_adu2wav_cluster_id, chose_text2sem_cluster_id) = load_draw.multi_view_filter_process(
                hparams,
                cluster_mean_array=cluster_mean_array,
                sem_type_list=sem_type_list,
                aud_cluster_mean_array=aud_cluster_mean_array,
                sample_label_array_cluster=sample_label_array_cluster,
                aud_sample_label_array_cluster=aud_sample_label_array_cluster,
                semantic_cls_fea_list=semantic_cls_fea_list,
                final_filter_aux_sample_dict=final_filter_aux_sample_dict,
                mulview_sf_number=hparams['mulview_sf_number'],
                mulview_filter_mode=hparams['mulview_filter_mode'],
                load_precal_multi_view_clusters=hparams['load_precal_multi_view_clusters'],
                mulview_sf_weights=hparams['mulview_sf_weights']
            )
            print('the min balanced sample number of any clusters is, ', balanced_min_sample_num_in_all_clustets)
            print('the total balanced sample number of all clusters is, ', len(balanced_selected_sample_dataset_id))

            # draw multi-view SM filter res distribution
            balanced_fin_filter_aux_sample_dict = {
                'text2transcript_cluster_id' : chose_text2transcript_cluster_id,
                'aud2wav_cluster_id': chose_adu2wav_cluster_id,
                'text2sem_cluster_id': chose_text2sem_cluster_id,
            }
            (balanced_total_gt_div_score, balanced_total_pr_div_score, _cluster_mean_array, _sem_type_list, _aud_cluster_mean_array,
            _sample_label_array_cluster, _aud_sample_label_array_cluster, _semantic_cls_fea_list, _balanced_fin_filter_aux_sample_dict) = load_draw.load_draw(
                hparams,
                final_filter_aux_sample_dict=balanced_fin_filter_aux_sample_dict,
                T2_transcript_cluster_res_path=hparams['cluster_res_path'],
                Aux_aud_cluster_res_path=hparams['aud_cluster_res_path'],
                T2_semantics_cluster_res_path=hparams['semantic2vec_fea_file'],
                diagram_save_folder=diagram_save_folder,
                sf_type='balance'  # 'balance', None
            )
            # balanced_total_gt_div_score is the entropy of an uniform reference in ablation study
            # balanced_total_pr_div_score is the entropy of chosen method (reported entropy in Table 4)
            print(f'multi-view sf has total gt_div_score and total pr_div_score as {balanced_total_gt_div_score} and {balanced_total_pr_div_score}.')

    # save/load the MCSS results
    if 'multi_view_balance' in hparams.keys() and hparams['multi_view_balance']:
        if hparams['recal_filter_csv']:
            selected_aux_cvs = prepare_process_data.filter_csv(hparams['merge_save_path'], balanced_selected_sample_dataset_id,
                                                               hparams['seleted_aux_csv_wo'])
            print('save multi-view filter_csv to: ', hparams['seleted_aux_csv_wo'])
        else:
            print('load multi-view filter_csv from: ', hparams['seleted_aux_csv_wo'])



    ### 4. process to get synthetic semantics (labels)
    print('------------------------------------------------')
    print('4. process to get synthetic labels')
    if "use_selectnet" not in hparams.keys():
        hparams["use_selectnet"] = False

    syn_save_path = hparams['filter_res_by_text_similary_dir']
    syn_save_name = hparams["mid_syn_label_file"]
    syn_save_file = os.path.join(syn_save_path, syn_save_name)
    if hparams["cal_syn_label"] == True or hparams["use_selectnet"] == True:
        ## 4.0 load/re-train a model (\Theta^t_{T->S} trained by D^t_{T->S})
        # Brain class initialization
        nlu_hparams_file = sys.argv[-1]
        with open(nlu_hparams_file) as nlu_fin:
            nlu_hparams = load_hyperpyyaml(nlu_fin)  # include the processes to load modules

        run_on_main(nlu_hparams["pretrainer"].collect_files)
        nlu_hparams["pretrainer"].load_collected(device=run_opts["device"])

        nlu_brain = NLU_text2sem.NLU(
            modules=nlu_hparams["modules"],
            opt_class=nlu_hparams["opt_class"],
            hparams=nlu_hparams,
            run_opts=run_opts,
            checkpointer=nlu_hparams["checkpointer"],
        )

        if hparams["use_selectnet"] == True and nlu_hparams['train_nlu']==False:
            nlu_brain.on_evaluate_start() # test whether the pre-trained model has been loaded, it is embedded in nlu_brain.evaluate(...)

    if hparams['use_nlu_bert_enc'] == True:
        hparams['nlu_tokenizer_type'] = 'bert-base-uncased' #nlu_hparams['slu_enc'].model_name

    if hparams["cal_syn_label"] == True:

        ## split the T2 (D^t_{T->S}) into train-val-test datasets
        prepare_process_data.split_csv_ratio(
            source_name=hparams['vs_csv_train'],
            target_name1=hparams['tr_vs_csv_train'],
            target_name2=hparams['dev_vs_csv_train'],
            target_name3=hparams['te_vs_csv_train'],
            split_ratio=hparams['sel_aux_csv_wi_ratio']
        )

        if hparams['data_name'] == 'slurp': # MiniPS2SLURP
            (tr_vs_csv_train, dev_vs_csv_train, te_vs_csv_train, asr_tokenizer, slu_tokenizer) \
            = slurp_process.dataio_prepare_nlu(nlu_hparams, train_csv=hparams['tr_vs_csv_train'],
                                               dev_csv=hparams['dev_vs_csv_train'],
                                               test_csv=hparams['te_vs_csv_train'])
        elif hparams['data_name'] == 'slue-voxpopuli': # VoxPopuli2SLUE
            (tr_vs_csv_train, dev_vs_csv_train, te_vs_csv_train, asr_tokenizer, slu_tokenizer) \
            = slue_voxpopuli_process.dataio_prepare_nlu(nlu_hparams, train_csv=hparams['tr_vs_csv_train'],
                                               dev_csv=hparams['dev_vs_csv_train'],
                                               test_csv=hparams['te_vs_csv_train'])
        else:
            raise ValueError('data_name is wrongly set')


        selected_train_data = \
            merge_process.dataio_prepare_nlu(hparams, tar_name1=hparams['seleted_aux_csv_wo'], data_role='aux',
                                             nlu_hparams=nlu_hparams)
        print('selected data is processed for generating synthetic labels')

        nlu_brain.slu_tokenizer = slu_tokenizer  # This tokenizer is used for the semantics
        if nlu_hparams['train_nlu']:
            print('start train NLU')
            nlu_brain.fit(  # it will auto restart with the nearest epoch if the setting are original.
                nlu_brain.hparams.epoch_counter,
                tr_vs_csv_train, # tr_vs_csv_train, # vs_train_data, # should use their text data
                dev_vs_csv_train, # valid_set,
                train_loader_kwargs=nlu_hparams["dataloader_opts"],
                valid_loader_kwargs=nlu_hparams["dataloader_opts"],
            )
            print('finished training of NLU')
        else:
            print('skip NLU training')

        # test nlu_brain
        if 'eval_nlu' in hparams.keys() and hparams['eval_nlu']:
            print("start evaluate trained/loaded nlu model on the gb test set")
            if hparams['data_name'] == 'slurp':
                nlu_id_to_file = {}
                df = pd.read_csv(hparams["csv_test"])
                for i in range(len(df)):
                    nlu_id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]
            elif hparams['data_name'] == 'slue-voxpopuli':
                nlu_id_to_file = None
            syn_nlu_te_name = 'nlu_gb_inference.jsonl'
            syn_nlu_te_file = os.path.join(syn_save_path, syn_nlu_te_name)
            prepare_aux_text_embedding.delete_file(syn_nlu_te_file)
            nlu_test_data = \
                merge_process.dataio_prepare_nlu(hparams, tar_name1=hparams['csv_test'], data_role='tar',
                                                 nlu_hparams=nlu_hparams)
            nlu_brain.infer_syn_label(
                nlu_test_data,
                nlu_hparams['slu_tokenizer'],  # need change in the bert
                id_to_file=nlu_id_to_file,
                save_path=syn_save_path,
                save_file=syn_nlu_te_name,
                test_loader_kwargs=nlu_hparams["dataloader_opts"]
            )
            # 'gb' is an global divided test set for inference (the uniform test used in all set),
            print('the nlu eval res on gb_test_set is in, ', syn_nlu_te_file)


        ## Step 4.1 infer the semantics for seleted auxiliary dataset and ## Step 4.2 save the synthetic semantics
        id_to_file=None

        print('start infer synthetic labels')
        prepare_aux_text_embedding.delete_file(syn_save_file)
        nlu_brain.infer_syn_label(
            selected_train_data,
            nlu_hparams['slu_tokenizer'], # need change in the bert
            id_to_file=id_to_file,
            save_path=syn_save_path,
            save_file=syn_save_name,
            test_loader_kwargs=nlu_hparams["dataloader_opts"]
        )
    else:
        print('skip infer synthetic labels')
        if not os.path.exists(syn_save_file):
            print(f"{syn_save_file} does not exist, please set cal_syn_label to True!")
    print(f"ids and synthetic labels are save in {syn_save_file}")

    print('NLU training and inferring synthetic semantics (labels) have been finsihed')



    ### 5. process to train S3
    print('------------------------------------------------')
    print('5. process to train S3')
    ## 5.1 load the synthetic semantics & merge the audio and synthetic info into one csv
    if hparams['data_name'] == 'slurp':
        prepare_process_data.insert_lable_slurp(ini_sel_file=hparams['seleted_aux_csv_wo'], syn_save_file=syn_save_file,
                                       insert_sel_file=hparams['seleted_aux_csv_wi'])
    elif hparams['data_name'] == 'slue-voxpopuli':
        prepare_process_data.insert_lable_slue(ini_sel_file=hparams['seleted_aux_csv_wo'], syn_save_file=syn_save_file,
                                                insert_sel_file=hparams['seleted_aux_csv_wi'])

    ## 5.2 split the synthetic semantic csv (D_{A->T} + respective synthetic semantics) into train-val-test datasets
    prepare_process_data.split_csv_ratio(
        source_name=hparams['seleted_aux_csv_wi'],
        target_name1=hparams['tr_sel_aux_csv_wi'],
        target_name2=hparams['dev_sel_aux_csv_wi'],
        target_name3=hparams['te_sel_aux_csv_wi'],
        split_ratio=hparams['sel_aux_csv_wi_ratio']
    )

    # here we create the datasets objects as well as tokenization
    (sel_aux_train_set, sel_aux_valid_set, sel_aux_lc_test_set, slu_tokenizer_direct, nlu_tokenizer_transcript) = \
        merge_process.dataio_prepare_asu_lc(hparams,
                                            target_name1=hparams['tr_sel_aux_csv_wi'],
                                            target_name2=hparams['dev_sel_aux_csv_wi'],
                                            target_name3=hparams['te_sel_aux_csv_wi'],
                                            )

    ## use audio & synthetic semantics to train the S3 (\Tilde{\Theta}_{A->S})
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Move the wav2vec2
    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # freeze the feature extractor part when unfreezing # hparams["wav2vec2"].model.feature_extractor constructs the features from raw audio waveform
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Brain class initialization
    if hparams["use_selectnet"] == False:
        print("use basic version of SLU, no selective_net")
        slu_brain = direct_audio2sem.SLU(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        assert "cm_sel_net" not in slu_brain.modules.keys()
    else:
        ### below is the cross-modal selective-net (CMSN) in Sec
        print("use advanced version of SLU, with selective_net")
        slu_brain = sel_direct_audio2sem.SLU(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        slu_brain.transcript_tokenizer = nlu_tokenizer_transcript # used for transcript
        assert "cm_sel_net" in slu_brain.modules.keys()

    # adding objects to trainer:
    slu_brain.tokenizer = slu_tokenizer_direct  # used for semantics


    # Training S3 (\tilde{\Theta}_{A->S})
    if hparams['retrain_s3']:
        print('start train S3 (tilde{\Theta}_{A->S})')
        if hparams["use_selectnet"] == False:
            slu_brain.fit(
                slu_brain.hparams.epoch_counter,
                sel_aux_train_set, # sel_aux_train_set,
                sel_aux_valid_set, # sel_aux_valid_set
                train_loader_kwargs=hparams["dataloader_opts"],
                valid_loader_kwargs=hparams["dataloader_opts"],
            )
        else:
            sel_train_valid_set_text_fea_dict = {}

            # cal the trainscript in advance to save GPU memory
            print("start save pre-calculated nlu encoder embeddings for ASR texts")
            nlu_fea_dict = nlu_brain.infer_save_text_fea([sel_aux_train_set, sel_aux_valid_set])
            # nlu_fea_dict = nlu_brain.infer_save_text_fea([sel_aux_valid_set])
            del nlu_brain
            torch.cuda.empty_cache()

            print("finished text fea saving, strart train S3 (tilde{\Theta}_{A->S})")
            slu_brain.fit(
                slu_brain.hparams.epoch_counter,
                sel_aux_train_set,  # sel_aux_train_set,
                sel_aux_valid_set,  # sel_aux_valid_set,
                nlu=nlu_fea_dict,
                train_loader_kwargs=hparams["dataloader_opts"],
                valid_loader_kwargs=hparams["dataloader_opts"],
            )

    # 'lc' is an local divided test set for inference (not the uniform test used in all set),
    # thus <cal_lc_infer> can be skipped by setting it False in the yaml file
    if hparams['cal_lc_infer']:
        ### 6. evaluate S3 on lc
        print('------------------------------------------------')
        print('6. evaluate S3 on lc')
        print("Creating id_to_file mapping...")
        if hparams['data_name'] == 'slurp':
            lc_id_to_file = {}
            df = pd.read_csv(hparams["te_sel_aux_csv_wi"])
            for i in range(len(df)):
                lc_id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]
        elif hparams['data_name'] == 'slue-voxpopuli':
            lc_id_to_file = None

        slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
        print('start evaluate S3 on lc')
        prepare_aux_text_embedding.delete_file(os.path.join(hparams["filter_res_by_text_similary_dir"], hparams["lc_infer_res"]))
        slu_brain.infer_objectives_loss(
            sel_aux_lc_test_set,
            save_path=hparams["filter_res_by_text_similary_dir"],
            save_name=hparams["lc_infer_res"],
            id_to_file=lc_id_to_file,
            test_loader_kwargs=hparams["dataloader_opts"]
        )
        print('finished lc test and save res at, ', os.path.join(hparams["filter_res_by_text_similary_dir"], hparams["lc_infer_res"]))

    # 'gb' is an global divided test set for inference (the uniform test used in all set),
    # thus <cal_lc_infer> should be always conducted by setting it True in the yaml file, if we want the test results
    if hparams['cal_gb_infer']:
        ### 6. evaluate S3 on gb
        print('------------------------------------------------')
        print('6. evaluate S3 on gb')
        # _slu_tokenizer should be equal to slu_tokenizer
        (gb_test_set, _slu_tokenizer,) = merge_process.dataio_prepare_asu_gb(hparams, tar_name1=hparams["csv_test"])
        if hparams['data_name'] == 'slurp':
            print("Creating id_to_file mapping...")
            gb_id_to_file = {}
            df = pd.read_csv(hparams["csv_test"])
            for i in range(len(df)):
                gb_id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]
        elif hparams['data_name'] == 'slue-voxpopuli':
                gb_id_to_file = None

        slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
        print('start evaluate S3 on gc')
        prepare_aux_text_embedding.delete_file(
            os.path.join(hparams["filter_res_by_text_similary_dir"], hparams["gb_infer_res"]))
        slu_brain.infer_objectives_loss(
            gb_test_set,
            save_path=hparams["filter_res_by_text_similary_dir"],
            save_name=hparams["gb_infer_res"],
            id_to_file=gb_id_to_file,
            test_loader_kwargs=hparams["dataloader_opts"]
        )
        print('finished gb test and save res at, ', os.path.join(hparams["filter_res_by_text_similary_dir"], hparams["gb_infer_res"]))

    print('Cross-Modal Selective Self-Training (CMSST) Finished!')



