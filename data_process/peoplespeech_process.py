import torch
import speechbrain as sb


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

# def dataio_prepare_nlu(hparams, data_role, nlu_hparams=None):
#     # mode = "all": consider the train, dev and test
#     #         "train": only consider the train set, used for the filtered csv
#     #         "train" will 1. only output train set 2. add flac info into the output types
#     """This function prepares the datasets to be used in the brain class.
#     It also defines the data processing pipeline through user-defined functions."""
#     role_prefix = ""
#     if data_role == 'aux':
#         role_prefix = "aux_"
#     data_folder = hparams[role_prefix+"data_folder"]
#
#
#
#     train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
#             csv_path=hparams['seleted_aux_csv_wo'], replacements={"data_root": data_folder},
#         )
#
#
#
#     if hparams[role_prefix+"sorting"] == "ascending":
#         # we sort training data to speed up training and get better results.
#         train_data = train_data.filtered_sorted(sort_key="duration")
#         # when sorting do not shuffle in dataloader ! otherwise is pointless
#         hparams[role_prefix+"dataloader_opts"]["shuffle"] = False
#
#     elif hparams[role_prefix+"sorting"] == "descending":
#         train_data = train_data.filtered_sorted(
#             sort_key="duration", reverse=True
#         )
#         # when sorting do not shuffle in dataloader ! otherwise is pointless
#         hparams[role_prefix+"dataloader_opts"]["shuffle"] = False
#
#     elif hparams[role_prefix+"sorting"] == "random":
#         pass
#
#     else:
#         raise NotImplementedError(
#             "sorting must be random, ascending or descending"
#         )
#
#
#     datasets = [train_data]
#
#     tokenizer = hparams[role_prefix+"tokenizer"]
#
#     asr_tokenizer = nlu_hparams["asr_tokenizer"]
#     slu_tokenizer = nlu_hparams["slu_tokenizer"]
#
#     # 2. Define audio pipeline:
#     @sb.utils.data_pipeline.takes("wav", "ori_source")
#     @sb.utils.data_pipeline.provides("sig")
#     def audio_pipeline(wav, ori_source):
#         if ori_source == "peoplespeech":
#             path = hparams['peoplespeech_data_folder'] + '/subset' + wav
#         elif ori_source == "slurp":
#             path = hparams['slurp_data_folder'] + '/' + wav
#         sig = sb.dataio.dataio.read_audio(path)
#         return sig
#
#     # if mode == "train_asu":
#     #     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal
#
#     @sb.utils.data_pipeline.takes("transcript")
#     @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
#     def transcript_pipeline(transcript):
#         transcript = transcript.upper()
#         yield transcript
#         transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
#         transcript_tokens = torch.LongTensor(transcript_tokens_list)
#         yield transcript_tokens
#
#
#     sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)
#
#     # 2. Define id pipeline: # unnecessary
#     # @sb.utils.data_pipeline.takes("ID")
#     # @sb.utils.data_pipeline.provides("ID")
#     # def id_pipeline(ID):
#     #     return ID
#
#     # sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline) # cal audio signal
#
#     # only can be used in asu
#     # # 3. Define text-semantics pipeline:
#     # @sb.utils.data_pipeline.takes("semantics")
#     # @sb.utils.data_pipeline.provides(
#     #     "semantics",
#     #     "semantics_token_list",
#     #     "semantics_tokens_bos",
#     #     "semantics_tokens_eos",
#     #     "semantics_tokens",
#     # )
#     # def semantics_pipeline(semantics):
#     #     yield semantics
#     #     semantics_tokens_list = slu_tokenizer.encode_as_ids(semantics)
#     #     yield semantics_tokens_list
#     #     semantics_tokens_bos = torch.LongTensor(
#     #         [hparams["bos_index"]] + (semantics_tokens_list)
#     #     )
#     #     yield semantics_tokens_bos
#     #     semantics_tokens_eos = torch.LongTensor(
#     #         semantics_tokens_list + [hparams["eos_index"]]
#     #     )
#     #     yield semantics_tokens_eos
#     #     semantics_tokens = torch.LongTensor(semantics_tokens_list)
#     #     yield semantics_tokens
#
#     # if mode == "train_asu":
#     #     sb.dataio.dataset.add_dynamic_item(datasets, semantics_pipeline)
#
#     # # 4. Define text-transcript pipeline:
#     # @sb.utils.data_pipeline.takes("text")
#     # @sb.utils.data_pipeline.provides("text")
#     # def transcript_pipeline(text):
#     #     yield text
#         # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
#         # transcript_tokens = torch.LongTensor(transcript_tokens_list)
#         # yield transcript_tokens
#
#     # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)
#
#
#     sb.dataio.dataset.set_output_keys(
#         datasets,
#         ["id", "transcript", "transcript_tokens"],
#         # ["id"],
#         # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
#     )
#
#     return train_data


def dataio_prepare_nlu(hparams, data_role, nlu_hparams=None):
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
            csv_path=hparams['seleted_aux_csv_wo'], replacements={"data_root": data_folder},
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

    tokenizer = hparams[role_prefix+"tokenizer"]

    asr_tokenizer = nlu_hparams["asr_tokenizer"]
    slu_tokenizer = nlu_hparams["slu_tokenizer"]

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

    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    def transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        transcript_tokens = torch.LongTensor(transcript_tokens_list)
        yield transcript_tokens


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


    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "transcript", "transcript_tokens"],
        # ["id"],
        # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )

    return train_data