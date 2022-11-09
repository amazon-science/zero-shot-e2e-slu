import json
import os
from speechbrain.utils.data_utils import download_file
import shutil
from tqdm.contrib import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
import re
import torchtext.vocab as vocab
import torch.nn as nn
import torch

from models import NLU_text2sem, direct_audio2sem, sel_direct_audio2sem
from speechbrain.utils.distributed import run_on_main
import torch.nn.functional as F

# def prepare_word2vec(hparams):
#     word2vec_path = hparams["word2vec_pretrained_path"]
#     word2vec_name = hparams["word2vec_name"]
#     save_path = os.path.join(word2vec_path, word2vec_name)
#     if word2vec_name in ['glove']:
#         zip_suffix = '.zip'
#         word2vec_matrixfile = 'glove.840B.300d.txt'
#     else:
#         print('The word2vec_name is wrong!')
#         raise ValueError
#     word2vec_zipfile = word2vec_name + zip_suffix
#     zip_location = os.path.join(save_path, word2vec_zipfile)
#     if os.path.exists(os.path.join(save_path, word2vec_matrixfile)):
#         print(f"The pretrained file of {word2vec_matrixfile} exist! SKIP download!")
#         return
#     else:
#         if not os.path.exists(zip_location):
#             if word2vec_name == 'glove':
#                 url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
#             download_file(url, zip_location, unpack=True)
#         else:
#             print(f'Extracting {zip_location}')
#             shutil.unpack_archive(zip_location, save_path)

def prepare_word2vec(hparams):
    word2vec_path = hparams["word2vec_pretrained_path"]
    word2vec_name = hparams["word2vec_name"]
    save_path = os.path.join(word2vec_path, word2vec_name)
    if 'glove' in word2vec_name:
        pretrained_word2vec = vocab.GloVe(name='840B', dim=300, cache=save_path)
    else:
        print('The word2vec_name is wrong!')
        raise ValueError
    return pretrained_word2vec


def clean_str(string):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def sent_parse(text, lemmatizer):
    sentences = nltk.tokenize.sent_tokenize(text.lower())
    sentences_tokens = []

    for sentence in sentences:
        tokens = nltk.tokenize.wordpunct_tokenize(sentence)
        processed_tokens = []
        for token in tokens:
            token = clean_str(token)
            token = lemmatizer.lemmatize(token)
            if not token:
                processed_tokens.append(token)
        sentences_tokens.append(tokens)
    return sentences_tokens


def cal_word2vec_on_csv(hparams, data_name, train_data, save_json_path, pre_trained_word2vec):
    nltk.download('omw-1.4')
    # if os.path.exists(save_json_path):
    #     print(f'{save_json_path} exists, skip calculating word2vec')
    #     return
    if data_name == 'slurp' or data_name == 'merge':
        id_key_name, text_key_name = 'id', 'transcript'
    elif data_name == 'librispeech':
        id_key_name, text_key_name = 'id', 'wrd'
    elif data_name == 'slue-voxpopuli' or data_name == 'slue-voxpopuli-full':
        id_key_name, text_key_name = 'id', 'transcript' # 'normalized_text'
    elif data_name == 'peoplespeech':
        id_key_name, text_key_name = 'id', 'text'
    else:
        print('data_name is invalid!')
        raise ValueError

    lemmatizer = WordNetLemmatizer()
    vocab_size, embed_dim = pre_trained_word2vec.vectors.shape
    word_embedding = nn.Embedding(vocab_size+1, embed_dim)
    word_embedding.weight.data.copy_(torch.cat((pre_trained_word2vec.vectors.data, torch.zeros(1, embed_dim)), dim=0))

    word_embedding = word_embedding.cuda()
    # voc_list = pre_trained_word2vec.stoi.keys()

    print("Pretrained embedding weights loaded ...")

    with open(save_json_path, 'a+', encoding='utf-8') as f:
        with tqdm(
                train_data,
                initial=0,
                dynamic_ncols=True,
                disable=False,
        ) as t:
            for batch in t:
                row_dict = {}
                row_dict['id'] = batch[id_key_name] # there is no shuffle in default train_data sample
                row_dict['transcript'] = batch[text_key_name]
                mid_sent_res = sent_parse(batch[text_key_name], lemmatizer)
                if len(mid_sent_res) == 0:
                    continue
                row_dict['token_text'] = mid_sent_res[0]
                row_dict['token_id'] = []
                for wrd in row_dict['token_text']:
                    try:
                        row_dict['token_id'].append(pre_trained_word2vec.stoi[wrd])
                    except:
                        # use unknown token: the last token
                        row_dict['token_id'].append(vocab_size)

                row_dict[hparams["word2vec_name"] + '_fea_vec'] = word_embedding(torch.LongTensor(row_dict['token_id']).cuda())
                row_dict[hparams["word2vec_name"] + '_mean_fea_vec'] = torch.mean(row_dict[hparams["word2vec_name"] + '_fea_vec'], dim=0)
                row_dict[hparams["word2vec_name"] + '_std_fea_vec'] = torch.std(row_dict[hparams["word2vec_name"] + '_fea_vec'], dim=0)

                # row_dict[hparams["word2vec_name"] + '_fea_vec'] = row_dict[hparams["word2vec_name"] + '_fea_vec'].cpu().tolist()
                row_dict.pop(hparams["word2vec_name"] + '_fea_vec')
                row_dict[hparams["word2vec_name"] + '_mean_fea_vec'] = row_dict[hparams["word2vec_name"] + '_mean_fea_vec'].cpu().tolist()
                row_dict[hparams["word2vec_name"] + '_std_fea_vec'] = row_dict[hparams["word2vec_name"] + '_std_fea_vec'].cpu().tolist()

                json.dump(row_dict, f)
                f.write('\n')


        print(f'finished word2vec {data_name}!')

def torch_cov(input_vec):
    x = input_vec- torch.mean(input_vec,dim=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def cal_semantic2vec_on_csv_indiviual_cov(hparams, data_name, train_data, save_json_path, pre_trained_word2vec):
    nltk.download('omw-1.4')


    if data_name == 'slurp' or data_name == 'slue-voxpopuli':
        id_key_name = 'id'
        semantic_key_name = 'semantics'
    else:
        print('data_name is invalid!')
        raise ValueError

    lemmatizer = WordNetLemmatizer()
    vocab_size, embed_dim = pre_trained_word2vec.vectors.shape
    word_embedding = nn.Embedding(vocab_size+1, embed_dim)
    word_embedding.weight.data.copy_(torch.cat((pre_trained_word2vec.vectors.data, torch.zeros(1, embed_dim)), dim=0))

    word_embedding = word_embedding.cuda()
    # voc_list = pre_trained_word2vec.stoi.keys()

    print("Pretrained embedding weights loaded ...")

    semantics_dict = {}
    with open(save_json_path, 'a+', encoding='utf-8') as f:
        with tqdm(
                train_data,
                initial=0,
                dynamic_ncols=True,
                disable=False,
        ) as t:
            for batch in t:

                if data_name == 'slurp':
                    mid_semantics = batch[semantic_key_name]['entities']
                elif data_name == 'slue-voxpopuli':
                    mid_semantics = batch[semantic_key_name]
                mid_semantics = eval(mid_semantics.replace('|', ','))
                if len(mid_semantics) == 0:
                    continue
                else:
                    for sub_semantic in mid_semantics:
                        if sub_semantic['type'] not in semantics_dict.keys():
                            semantics_dict[sub_semantic['type']] = [sub_semantic['filler']]
                        else:
                            semantics_dict[sub_semantic['type']].append(sub_semantic['filler'])

        cur_key_list = list(semantics_dict.keys())
        # calculate gaussian distribution
        for sub_type in cur_key_list:
            # semantics_dict[sub_type] = list(set(semantics_dict[sub_type]))
            mid_sub_type_res = []
            for ele_text in semantics_dict[sub_type]:
                mid_sub_type_res.append(sent_parse(ele_text, lemmatizer)[0])

            # reduce repeated semantics
            mid_sub_type_res_str = [str(str_list) for str_list in mid_sub_type_res]
            mid_sub_type_res_str = set(mid_sub_type_res_str)
            mid_sub_type_res = [eval(str_list) for str_list in mid_sub_type_res_str]

            semantic_token_id = sub_type+'token_id'
            semantic_fea_vec = hparams["word2vec_name"] + '_' + sub_type  + '_fea_vec'
            semantic_mean_fea_vec = hparams["word2vec_name"] + '_' + sub_type  + '_mean_fea_vec'
            semantic_cov_mat = hparams["word2vec_name"] + '_' + sub_type  + '_cov_mat'
            semantic_cov_mat_inv = hparams["word2vec_name"] + '_' + sub_type  + '_cov_mat_inv'

            semantics_dict[semantic_token_id] = []
            semantics_dict[semantic_fea_vec] = []

            # for wrd_list in semantics_dict[sub_type]:
            for wrd_list in mid_sub_type_res:
                mid_text_ids = []
                for wrd in wrd_list:
                    try:
                        mid_text_ids.append(pre_trained_word2vec.stoi[wrd])
                    except:
                        # use unknown token: the last token
                        mid_text_ids.append(vocab_size)

                semantics_dict[semantic_token_id].append(mid_text_ids)
                sin_sample_fea = word_embedding(torch.LongTensor(mid_text_ids).cuda())
                sin_sample_fea_mean = torch.mean(sin_sample_fea, dim=0)
                semantics_dict[semantic_fea_vec].append(sin_sample_fea_mean.unsqueeze(0))

            # cal gau for a subtype word2vec features
            semantics_dict[semantic_fea_vec] = torch.cat(semantics_dict[semantic_fea_vec], dim=0)
            semantics_dict[semantic_mean_fea_vec] = torch.mean(semantics_dict[semantic_fea_vec], dim=0)
            semantics_dict[semantic_cov_mat] = torch_cov(semantics_dict[semantic_fea_vec])
            semantics_dict[semantic_cov_mat_inv] = torch.linalg.inv(semantics_dict[semantic_cov_mat])

            # row_dict[hparams["word2vec_name"] + '_fea_vec'] = row_dict[hparams["word2vec_name"] + '_fea_vec'].cpu().tolist()
            semantics_dict.pop(semantic_token_id)
            semantics_dict.pop(semantic_fea_vec)
            semantics_dict.pop(semantic_cov_mat)
            semantics_dict[semantic_mean_fea_vec] = semantics_dict[semantic_mean_fea_vec].cpu().tolist()
            semantics_dict[semantic_cov_mat_inv] = semantics_dict[semantic_cov_mat_inv].cpu().tolist()

        print(semantics_dict.keys())
        semantics_dict['semantic_keywords'] = cur_key_list

        del sin_sample_fea
        del sin_sample_fea_mean

        json.dump(semantics_dict, f)
        f.write('\n')

        print(f'finished semantic2vec {data_name}!')

def cal_semantic2vec_on_csv_common_cov(hparams, data_name, train_data, save_json_path, pre_trained_word2vec):
    nltk.download('omw-1.4')


    if data_name == 'slurp' or data_name == 'slue-voxpopuli':
        id_key_name = 'id'
        semantic_key_name = 'semantics'
    else:
        print('data_name is invalid!')
        raise ValueError

    lemmatizer = WordNetLemmatizer()
    vocab_size, embed_dim = pre_trained_word2vec.vectors.shape
    word_embedding = nn.Embedding(vocab_size+1, embed_dim)
    word_embedding.weight.data.copy_(torch.cat((pre_trained_word2vec.vectors.data, torch.zeros(1, embed_dim)), dim=0))

    word_embedding = word_embedding.cuda()
    # voc_list = pre_trained_word2vec.stoi.keys()

    print("Pretrained embedding weights loaded ...")

    semantics_dict = {}
    sample_semantic_list = []
    with open(save_json_path, 'a+', encoding='utf-8') as f:
        with tqdm(
                train_data,
                initial=0,
                dynamic_ncols=True,
                disable=False,
        ) as t:
            for batch in t:

                if data_name == 'slurp':
                    mid_semantics = batch[semantic_key_name]
                    mid_semantics = eval(mid_semantics.replace('|', ','))
                    mid_semantics = mid_semantics["entities"]
                elif data_name == 'slue-voxpopuli':
                    mid_semantics = batch[semantic_key_name]
                    mid_semantics = eval(mid_semantics.replace('|', ','))
                if len(mid_semantics) == 0:
                    continue
                else:
                    for sub_semantic in mid_semantics:
                        sample_semantic_list.append(sub_semantic)
                        if sub_semantic['type'] not in semantics_dict.keys():
                            semantics_dict[sub_semantic['type']] = [sub_semantic['filler']]
                        else:
                            semantics_dict[sub_semantic['type']].append(sub_semantic['filler'])

        cur_key_list = list(semantics_dict.keys())

        semantic_comb_cov_mat = hparams["word2vec_name"] + '_' + 'combined' + '_cov_mat'
        semantic_comb_cov_mat_inv = hparams["word2vec_name"] + '_' + "combined" + '_cov_mat_inv'
        total_sample_fea_list = []

        # calculate gaussian distribution
        for sub_type in cur_key_list:
            # semantics_dict[sub_type] = list(set(semantics_dict[sub_type]))
            mid_sub_type_res = []
            for ele_text in semantics_dict[sub_type]:
                mid_sub_type_res.append(sent_parse(ele_text, lemmatizer)[0])

            # reduce repeated semantics
            mid_sub_type_res_str = [str(str_list) for str_list in mid_sub_type_res]
            mid_sub_type_res_str = set(mid_sub_type_res_str)
            mid_sub_type_res = [eval(str_list) for str_list in mid_sub_type_res_str]

            semantic_token_id = sub_type+'token_id'
            semantic_fea_vec = hparams["word2vec_name"] + '_' + sub_type  + '_fea_vec'
            semantic_mean_fea_vec = hparams["word2vec_name"] + '_' + sub_type  + '_mean_fea_vec'
            semantic_cov_mat = hparams["word2vec_name"] + '_' + sub_type  + '_cov_mat'
            semantic_cov_mat_inv = hparams["word2vec_name"] + '_' + sub_type  + '_cov_mat_inv'

            semantics_dict[semantic_token_id] = []
            semantics_dict[semantic_fea_vec] = []

            # for wrd_list in semantics_dict[sub_type]:
            for wrd_list in mid_sub_type_res:
                mid_text_ids = []
                for wrd in wrd_list:
                    try:
                        mid_text_ids.append(pre_trained_word2vec.stoi[wrd])
                    except:
                        # use unknown token: the last token
                        mid_text_ids.append(vocab_size)

                semantics_dict[semantic_token_id].append(mid_text_ids)
                sin_sample_fea = word_embedding(torch.LongTensor(mid_text_ids).cuda())
                sin_sample_fea_mean = torch.mean(sin_sample_fea, dim=0)
                semantics_dict[semantic_fea_vec].append(sin_sample_fea_mean.unsqueeze(0))
                total_sample_fea_list.append(sin_sample_fea_mean.unsqueeze(0))

            # cal gau for a subtype word2vec features
            semantics_dict[semantic_fea_vec] = torch.cat(semantics_dict[semantic_fea_vec], dim=0)

            semantics_dict[semantic_mean_fea_vec] = torch.mean(semantics_dict[semantic_fea_vec], dim=0)
            semantics_dict[semantic_cov_mat] = torch_cov(semantics_dict[semantic_fea_vec])
            semantics_dict[semantic_cov_mat_inv] = torch.linalg.inv(semantics_dict[semantic_cov_mat])

            # row_dict[hparams["word2vec_name"] + '_fea_vec'] = row_dict[hparams["word2vec_name"] + '_fea_vec'].cpu().tolist()
            semantics_dict.pop(semantic_token_id)
            semantics_dict.pop(semantic_fea_vec)
            semantics_dict.pop(semantic_cov_mat)
            semantics_dict[semantic_mean_fea_vec] = semantics_dict[semantic_mean_fea_vec].cpu().tolist()
            semantics_dict[semantic_cov_mat_inv] = semantics_dict[semantic_cov_mat_inv].cpu().tolist()

        # cal_total_cov & cov_inv
        total_sample_fea_tensors = torch.cat(total_sample_fea_list, dim=0)
        semantics_dict[semantic_comb_cov_mat] = torch_cov(total_sample_fea_tensors)
        semantics_dict[semantic_comb_cov_mat_inv] = torch.linalg.inv(semantics_dict[semantic_comb_cov_mat])
        semantics_dict[semantic_comb_cov_mat] = semantics_dict[semantic_comb_cov_mat].cpu().tolist()
        semantics_dict[semantic_comb_cov_mat_inv] = semantics_dict[semantic_comb_cov_mat_inv].cpu().tolist()


        print(semantics_dict.keys())
        semantics_dict['semantic_keywords'] = cur_key_list
        semantics_dict['sample_semantic_list'] = sample_semantic_list

        del sin_sample_fea
        del sin_sample_fea_mean
        del total_sample_fea_list

        json.dump(semantics_dict, f)
        f.write('\n')

        print(f'finished semantic2vec {data_name}!')

def extract_part_csv(target_csv_file, selected_sample_dict):
    pass

def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def cal_aud2vec_on_csv(hparams, data_name, train_data, save_json_path, pretrained_model):
    # nltk.download('omw-1.4')
    # if os.path.exists(save_json_path):
    #     print(f'{save_json_path} exists, skip calculating word2vec')
    #     return

    id_key_name, audio_key_name = 'id', 'sig'

    # if data_name == 'slurp' or data_name == 'merge':
    #     id_key_name, audio_key_name = 'id', 'transcript'
    # elif data_name == 'librispeech':
    #     id_key_name, text_key_name = 'id', 'wrd'
    # elif data_name == 'slue-voxpopuli' or data_name == 'slue-voxpopuli-full':
    #     id_key_name, text_key_name = 'id', 'transcript' # 'normalized_text'
    # elif data_name == 'peoplespeech':
    #     id_key_name, text_key_name = 'id', 'text'
    # else:
    #     print('data_name is invalid!')
    #     raise ValueError

    # lemmatizer = WordNetLemmatizer()
    # vocab_size, embed_dim = pretrained_model.vectors.shape
    # word_embedding = nn.Embedding(vocab_size+1, embed_dim)
    # word_embedding.weight.data.copy_(torch.cat((pretrained_model.vectors.data, torch.zeros(1, embed_dim)), dim=0))
    #
    # word_embedding = word_embedding.cuda()
    ## voc_list = pretrained_model.stoi.keys()




    print("Pretrained HuBert weights loaded ...")

    with open(save_json_path, 'a+', encoding='utf-8') as f:
        with tqdm(
                train_data,
                initial=0,
                dynamic_ncols=True,
                disable=False,
        ) as t:
            for batch in t:
                row_dict = {}
                row_dict['id'] = batch[id_key_name] # there is no shuffle in default train_data sample
                row_dict['sig'] = batch[audio_key_name]

                # mid_sent_res = sent_parse(batch[text_key_name], lemmatizer)
                #
                # if len(mid_sent_res) == 0:
                #     continue
                # row_dict['token_text'] = mid_sent_res[0]
                # row_dict['token_id'] = []
                # for wrd in row_dict['token_text']:
                #     try:
                #         row_dict['token_id'].append(pretrained_model.stoi[wrd])
                #     except:
                #         # use unknown token: the last token
                #         row_dict['token_id'].append(vocab_size)

                # a process to use register_forward_hook
                # mid_aud_feas = []
                # row_dict['sig'] = F.layer_norm(row_dict['sig'], row_dict['sig'].shape)
                # def extract_hook(module, input, output):
                #     print('This is hook!')
                #     print(output.shape)
                #     mid_aud_feas.append(output.clone().detach)
                # handle = pretrained_model.modules.wav2vec2.model.feature_extractor.register_forward_hook(extract_hook)
                # yyy = pretrained_model.modules.wav2vec2.model(row_dict['sig'].unsqueeze(0).cuda())
                # handle.remove()

                row_dict['sig'] = F.layer_norm(row_dict['sig'], row_dict['sig'].shape)
                mid_aud_fea = pretrained_model.modules.wav2vec2.model.feature_extractor(row_dict['sig'].unsqueeze(0).cuda())
                mid_aud_fea = mid_aud_fea.transpose(2, 1).squeeze(0)
                # print(pretrained_model.modules.wav2vec2(row_dict['sig'].unsqueeze(0).cuda()).shape)

                row_dict['sig' + '_mean_fea_vec'] = torch.mean(mid_aud_fea, dim=0).cpu().tolist()
                row_dict.pop('sig')

                assert len(row_dict['sig' + '_mean_fea_vec']) == 512
                del mid_aud_fea

                json.dump(row_dict, f)
                f.write('\n')


        print(f'finished aud2vec {data_name}!')

def load_hubert(hparams, run_opts):
    ## use audio & synthetic semantics to train the S3
    tokenizer = hparams["tokenizer"]

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Move the wav2vec2
    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # freeze the feature extractor part when unfreezing # hparams["wav2vec2"].model.feature_extractor constructs the features from raw audio waveform
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()


    # Brain class initialization
    print("use a pretrained HuBert in SLU to extract shallow audio features")
    # this SLU is a basic SLU model with pre-trained HuBert, there is no cross-modal selective net in it.
    slu_brain = direct_audio2sem.SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    slu_brain.tokenizer = tokenizer
    return slu_brain