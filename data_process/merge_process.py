import torch
import speechbrain as sb
# from transformers import RobertaTokenizer
from transformers import BertTokenizer


def dataio_prepare(hparams, data_role, mode="select_aux_sample", nlu_hparams=None):
    # mode = "all": consider the train, dev and test
    #         "train": only consider the train set, used for the filtered csv
    #         "train" will 1. only output train set 2. add flac info into the output types
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    if hparams["divide_csv_train"] and hparams["data_name"] == 'peoplespeech':
        vs_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix + "vs_csv_train"], replacements={"data_root": data_folder},
        )
        invs_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix + "invs_csv_train"], replacements={"data_root": data_folder},
        )
    else:
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix+"csv_train"], replacements={"data_root": data_folder},
        )



    if hparams[role_prefix+"sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    if mode == "select_aux_sample":
        valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix+"csv_valid"], replacements={"data_root": data_folder},
        )
        # valid_data = valid_data.filtered_sorted(sort_key="duration")

        test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix+"csv_test"], replacements={"data_root": data_folder},
        )
        # test_data = test_data.filtered_sorted(sort_key="duration")

        if hparams["divide_csv_train"] and hparams["data_name"] == 'peoplespeech':
            datasets = [vs_train_data, invs_train_data, valid_data, test_data]
        else:
            datasets = [train_data, valid_data, test_data]
    elif mode in ["train_nlu", "train_asu"]:
        datasets = [train_data]

    tokenizer = hparams[role_prefix+"tokenizer"]

    if mode in ["train_nlu"]:
        asr_tokenizer = nlu_hparams["asr_tokenizer"]
        slu_tokenizer = nlu_hparams["slu_tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("ID", "split")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(ID, split):
        path = ID + split + ".ogg"
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    if mode == "train_asu":
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal

    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    def transcript_pipeline(text):
        transcript = text.upper()
        yield transcript
        transcript_tokens_list = asr_tokenizer.encode_as_ids(text)
        transcript_tokens = torch.LongTensor(transcript_tokens_list)
        yield transcript_tokens

    if mode == "train_nlu":
        sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 2. Define id pipeline: # unnecessary
    # @sb.utils.data_pipeline.takes("ID")
    # @sb.utils.data_pipeline.provides("ID")
    # def id_pipeline(ID):
    #     return ID

    # sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline) # cal audio signal

    # 3. Define text-semantics pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics",
        "semantics_token_list",
        "semantics_tokens_bos",
        "semantics_tokens_eos",
        "semantics_tokens",
    )
    def semantics_pipeline(semantics):
        yield semantics
        semantics_tokens_list = slu_tokenizer.encode_as_ids(semantics)
        yield semantics_tokens_list
        semantics_tokens_bos = torch.LongTensor(
            [hparams["bos_index"]] + (semantics_tokens_list)
        )
        yield semantics_tokens_bos
        semantics_tokens_eos = torch.LongTensor(
            semantics_tokens_list + [hparams["eos_index"]]
        )
        yield semantics_tokens_eos
        semantics_tokens = torch.LongTensor(semantics_tokens_list)
        yield semantics_tokens

    if mode == "train_asu":
        sb.dataio.dataset.add_dynamic_item(datasets, semantics_pipeline)

    # # 4. Define text-transcript pipeline:
    # @sb.utils.data_pipeline.takes("text")
    # @sb.utils.data_pipeline.provides("text")
    # def transcript_pipeline(text):
    #     yield text
        # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        # transcript_tokens = torch.LongTensor(transcript_tokens_list)
        # yield transcript_tokens

    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 4. Set output:
    if mode == "select_aux_sample":
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "text"],
            # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
        )
    elif mode == "train_nlu":
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "transcript", "transcript_tokens"],
            # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
        )

    if mode == "select_aux_sample":
        if hparams["divide_csv_train"] and hparams["data_name"] == 'peoplespeech':
            return vs_train_data, invs_train_data, valid_data, test_data, tokenizer
        else:
            return train_data, valid_data, test_data, tokenizer
    elif mode in ["train_nlu", "train_asr"]:
        return train_data

def dataio_prepare_text_match(hparams, target_name1, data_role):
    # mode = "all": consider the train, dev and test
    #         "train": only consider the train set, used for the filtered csv
    #         "train" will 1. only output train set 2. add flac info into the output types
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    # data_folder = hparams[role_prefix+"data_folder"]



    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=target_name1,
            # replacements={"data_root": data_folder},
        )



    if hparams[role_prefix+"sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )


    datasets = [train_data]

    # tokenizer = hparams[role_prefix+"tokenizer"]


    # asr_tokenizer = nlu_hparams["asr_tokenizer"]
    # slu_tokenizer = nlu_hparams["slu_tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "ori_source")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, ori_source):
        if ori_source == "peoplespeech":
            path = hparams['peoplespeech_data_folder'] + '/subset/' + wav
        elif ori_source == "slurp":
            path = hparams['slurp_data_folder'] + '/' + wav
        elif ori_source == 'slue-voxpopuli':
            # key_folder = 'fine-tune'
            file_end = '.ogg'
            path = hparams['slue_voxpoluli_folder'] + '/' + wav + file_end
        elif ori_source == 'slue-voxpopuli-full':
            key_folder = wav[0:4]
            file_end = '.ogg'
            path = hparams['slue_voxpoluli_full_folder'] + '/' + key_folder + '/' + wav + file_end
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # if mode == "train_asu":
    #     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal

    # @sb.utils.data_pipeline.takes("text")
    # @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    # def transcript_pipeline(text):
    #     text = text.upper()
    #     transcript = text
    #     yield transcript
    #     transcript_tokens_list = asr_tokenizer.encode_as_ids(text)
    #     transcript_tokens = torch.LongTensor(transcript_tokens_list)
    #     yield transcript_tokens


    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 2. Define id pipeline: # unnecessary
    # @sb.utils.data_pipeline.takes("ID")
    # @sb.utils.data_pipeline.provides("ID")
    # def id_pipeline(ID):
    #     return ID

    # sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline) # cal audio signal


    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "transcript", "ori_source", "sig"],
        # ["id"],
        # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )

    return train_data

def dataio_prepare_nlu(hparams, tar_name1, data_role, nlu_hparams=None):
    # mode = "all": consider the train, dev and test
    #         "train": only consider the train set, used for the filtered csv
    #         "train" will 1. only output train set 2. add flac info into the output types
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]



    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=tar_name1, replacements={"data_root": data_folder},
        )



    if hparams[role_prefix+"sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )


    datasets = [train_data]

    ## tokenizer = hparams[role_prefix+"tokenizer"]

    if hparams['use_nlu_bert_enc']:
        # asr_tokenizer = RobertaTokenizer.from_pretrained(hparams['nlu_tokenizer_type'])
        asr_tokenizer = BertTokenizer.from_pretrained(hparams['nlu_tokenizer_type'])
    else:
        asr_tokenizer = nlu_hparams["asr_tokenizer"] #### !!! should be nlu_hparams

    # asr_tokenizer = nlu_hparams["asr_tokenizer"]
    ## slu_tokenizer = nlu_hparams["slu_tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "ori_source")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, ori_source):
        if ori_source == "peoplespeech":
            path = hparams['peoplespeech_data_folder'] + '/subset' + wav
        elif ori_source == "slurp":
            path = hparams['slurp_data_folder'] + '/' + wav
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    # if mode == "train_asu":
    #     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal

    # @sb.utils.data_pipeline.takes("transcript")
    # @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    # def transcript_pipeline(transcript):
    #     transcript = transcript.upper()
    #     yield transcript
    #     transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
    #     transcript_tokens = torch.LongTensor(transcript_tokens_list)
    #     yield transcript_tokens
    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    def transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        transcript_tokens = torch.LongTensor(transcript_tokens_list)
        yield transcript_tokens

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids")
    def bert_transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        transcript_tokens_dict = asr_tokenizer(transcript, return_tensors='pt')
        if transcript_tokens_dict['input_ids'].shape[1] > 512:
            transcript_tokens_input_ids = transcript_tokens_dict['input_ids'].squeeze(0)[0:512]
            transcript_tokens_attention_mask = transcript_tokens_dict['attention_mask'].squeeze(0)[0:512]
            transcript_tokens_token_type_ids = transcript_tokens_dict['token_type_ids'].squeeze(0)[0:512]
        else:
            transcript_tokens_input_ids = transcript_tokens_dict['input_ids'].squeeze(0)
            transcript_tokens_attention_mask = transcript_tokens_dict['attention_mask'].squeeze(0)
            transcript_tokens_token_type_ids = transcript_tokens_dict['token_type_ids'].squeeze(0)
        yield transcript_tokens_input_ids
        yield transcript_tokens_attention_mask
        yield transcript_tokens_token_type_ids


    if hparams['use_nlu_bert_enc']:
        sb.dataio.dataset.add_dynamic_item(datasets, bert_transcript_pipeline)
    else:
        sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 2. Define id pipeline: # unnecessary
    # @sb.utils.data_pipeline.takes("ID")
    # @sb.utils.data_pipeline.provides("ID")
    # def id_pipeline(ID):
    #     return ID

    # sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline) # cal audio signal

    # only can be used in asu
    # # 3. Define text-semantics pipeline:
    # @sb.utils.data_pipeline.takes("semantics")
    # @sb.utils.data_pipeline.provides(
    #     "semantics",
    #     "semantics_token_list",
    #     "semantics_tokens_bos",
    #     "semantics_tokens_eos",
    #     "semantics_tokens",
    # )
    # def semantics_pipeline(semantics):
    #     yield semantics
    #     semantics_tokens_list = slu_tokenizer.encode_as_ids(semantics)
    #     yield semantics_tokens_list
    #     semantics_tokens_bos = torch.LongTensor(
    #         [hparams["bos_index"]] + (semantics_tokens_list)
    #     )
    #     yield semantics_tokens_bos
    #     semantics_tokens_eos = torch.LongTensor(
    #         semantics_tokens_list + [hparams["eos_index"]]
    #     )
    #     yield semantics_tokens_eos
    #     semantics_tokens = torch.LongTensor(semantics_tokens_list)
    #     yield semantics_tokens

    # if mode == "train_asu":
    #     sb.dataio.dataset.add_dynamic_item(datasets, semantics_pipeline)

    # # 4. Define text-transcript pipeline:
    # @sb.utils.data_pipeline.takes("text")
    # @sb.utils.data_pipeline.provides("text")
    # def transcript_pipeline(text):
    #     yield text
        # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        # transcript_tokens = torch.LongTensor(transcript_tokens_list)
        # yield transcript_tokens

    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    if hparams['use_nlu_bert_enc']:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "transcript", "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"],
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "transcript", "transcript_tokens"],
        )

    # sb.dataio.dataset.set_output_keys(
    #     datasets,
    #     ["id", "transcript", "transcript_tokens"],
    # )

    return train_data



def dataio_prepare_asu_lc(hparams, target_name1, target_name2, target_name3, data_role=None):
    ## this function processes the
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]


    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name1, replacements={"data_root": data_folder},
    )


    if hparams[role_prefix+"sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams[role_prefix+"dataloader_opts"]["shuffle"] = False

    elif hparams[role_prefix+"sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name2, replacements={"data_root": data_folder},
    )
    try:
        valid_data = valid_data.filtered_sorted(sort_key="duration")
    except:
        pass

    # used for testing whether ASU model is well trained
    lc_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name3, replacements={"data_root": data_folder},
    )
    try:
        lc_test_data = lc_test_data.filtered_sorted(sort_key="duration")
    except:
        pass

    # used for testing the performance of whole system on the original data
    # gb_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    #     csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    # )
    # gb_test_data = gb_test_data.filtered_sorted(sort_key="duration")


    datasets = [train_data, valid_data, lc_test_data]


    if hparams['use_nlu_bert_enc']:
        # asr_tokenizer = RobertaTokenizer.from_pretrained(hparams['nlu_tokenizer_type'])
        asr_tokenizer = BertTokenizer.from_pretrained(hparams['nlu_tokenizer_type'])
    else:
        asr_tokenizer = hparams["asr_tokenizer"]   # used for transcript
    # asr_tokenizer = hparams["asr_tokenizer"]   # used for transcript
    tokenizer = hparams[role_prefix+"tokenizer"] # used for semantics


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "ori_source")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, ori_source):
        if ori_source == "peoplespeech":
            path = hparams['peoplespeech_data_folder'] + '/subset/' + wav
        elif ori_source == "slurp":
            path = hparams['slurp_data_folder'] + '/' + wav
        elif ori_source == "slue-voxpopuli-full":
            key_folder = wav[0:4]
            file_end = '.ogg'
            path = hparams['slue_voxpoluli_full_folder'] + '/' + key_folder + '/' + wav + file_end
        elif ori_source == "slue-voxpopuli":
            file_end = '.ogg'
            # path = hparams['slue_voxpoluli_folder'] +  + key_folder + '/' + wav + file_end
            path = hparams['slue_voxpoluli_folder'] + '/' + wav + file_end
        sig = sb.dataio.dataio.read_audio(path)

        ### only used for extreme/random case in multi-view sample filter (random filter)
        # print('sig shape: ', sig.shape)
        # set_max_audio_length = 450000
        # if sig.shape[0] > set_max_audio_length:
        #     sig = sig[0:set_max_audio_length]
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text-semantics pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams[role_prefix+"bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams[role_prefix+"eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # # # 4. Define text-transcript pipeline:
    # @sb.utils.data_pipeline.takes("transcript")
    # @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    # def transcript_pipeline(transcript):
    #     transcript = transcript.upper()
    #     yield transcript
    #     transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
    #     transcript_tokens = torch.LongTensor(transcript_tokens_list)
    #     # print("transcript_tokens is in shape of ", transcript_tokens.shape)
    #     yield transcript_tokens
    #
    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    def transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        transcript_tokens = torch.LongTensor(transcript_tokens_list)
        yield transcript_tokens

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids")
    def bert_transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        transcript_tokens_dict = asr_tokenizer(transcript, return_tensors='pt')
        if transcript_tokens_dict['input_ids'].shape[1] > 512:
            transcript_tokens_input_ids = transcript_tokens_dict['input_ids'].squeeze(0)[0:512]
            transcript_tokens_attention_mask = transcript_tokens_dict['attention_mask'].squeeze(0)[0:512]
            transcript_tokens_token_type_ids = transcript_tokens_dict['token_type_ids'].squeeze(0)[0:512]
        else:
            transcript_tokens_input_ids = transcript_tokens_dict['input_ids'].squeeze(0)
            transcript_tokens_attention_mask = transcript_tokens_dict['attention_mask'].squeeze(0)
            transcript_tokens_token_type_ids = transcript_tokens_dict['token_type_ids'].squeeze(0)
        yield transcript_tokens_input_ids
        yield transcript_tokens_attention_mask
        yield transcript_tokens_token_type_ids


    if hparams['use_nlu_bert_enc']:
        sb.dataio.dataset.add_dynamic_item(datasets, bert_transcript_pipeline)
    else:
        sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)


    if hparams['use_nlu_bert_enc']:
        sb.dataio.dataset.set_output_keys(
            datasets,
            # ["id", "transcript", "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"],
            ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript", "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"],
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript", "transcript_tokens"],
        )


    # # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets,
    #     ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript", "transcript_tokens"],
    # )

    return train_data, valid_data, lc_test_data, tokenizer, asr_tokenizer


def dataio_prepare_asu_gb(hparams, tar_name1, data_role=None):
    ## this function processes the
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    # used for testing whether ASU model is well trained
    gb_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=tar_name1, replacements={"data_root": data_folder},
    )
    try:
        gb_test_data = gb_test_data.filtered_sorted(sort_key="duration")
    except:
        pass

    # used for testing the performance of whole system on the original data
    # gb_test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    #     csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    # )
    # gb_test_data = gb_test_data.filtered_sorted(sort_key="duration")


    datasets = [gb_test_data]

    tokenizer = hparams[role_prefix+"tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        if hparams['data_name'] == "slurp":
            path = hparams['slurp_data_folder'] + '/' + wav
        elif hparams['data_name'] == "slue-voxpopuli":
            file_key_word = ".ogg"
            path = hparams['slue_voxpoluli_folder'] + '/' + wav + file_key_word

        sig = sb.dataio.dataio.read_audio(path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text-semantics pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams[role_prefix+"bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams[role_prefix+"eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # # 4. Define text-transcript pipeline:
    # @sb.utils.data_pipeline.takes("transcript")
    # @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    # def transcript_pipeline(transcript):
    #     yield transcript
    #     # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
    #     # transcript_tokens = torch.LongTensor(transcript_tokens_list)
    #     # yield transcript_tokens
    #
    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )

    return gb_test_data, tokenizer