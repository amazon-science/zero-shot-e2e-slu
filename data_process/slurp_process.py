import torch
import speechbrain as sb
from transformers import BertTokenizer #, RobertaTokenizer


def dataio_prepare(hparams, data_role):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    if hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
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

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[role_prefix+"csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[role_prefix+"csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    if hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        datasets = [vs_train_data, invs_train_data, valid_data, test_data]
    else:
        datasets = [train_data, valid_data, test_data]

    tokenizer = hparams[role_prefix+"tokenizer"]


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
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

    # 4. Define text-transcript pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript")#, "transcript_tokens")
    def transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        # transcript_tokens = torch.LongTensor(transcript_tokens_list)
        # yield transcript_tokens

    sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )
    if hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        return vs_train_data, invs_train_data, valid_data, test_data, tokenizer
    else:
        return train_data, valid_data, test_data, tokenizer



def dataio_prepare_ratio(hparams, data_role, split_ratio):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    if split_ratio == None:
        print('the split_ratio setting is wrong')
        raise ValueError
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    if hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        assert role_prefix == ""
        vs_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix + "vs_csv_train"], replacements={"data_root": data_folder},
        )
        invs_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix + "invs_csv_train"], replacements={"data_root": data_folder},
        )
        fine_tune_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[role_prefix + "fine_turn_csv"], replacements={"data_root": data_folder},
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

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[role_prefix+"csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[role_prefix+"csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    if hparams["split_ratio"] != "None" and hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        datasets = [vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data]
    elif hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        datasets = [vs_train_data, invs_train_data, valid_data, test_data]
    else:
        datasets = [train_data, valid_data, test_data]

    tokenizer = hparams[role_prefix+"tokenizer"]


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        wav = hparams['slurp_data_folder'] + '/' + wav
        sig = sb.dataio.dataio.read_audio(wav)
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

    # 4. Define text-transcript pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript")
    def transcript_pipeline(transcript):
        transcript = transcript.upper()
        yield transcript
        # transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        # transcript_tokens = torch.LongTensor(transcript_tokens_list)
        # yield transcript_tokens

    # sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )
    if hparams["split_ratio"] != "None" and hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        return vs_train_data, invs_train_data, fine_tune_data, valid_data, test_data, tokenizer
    elif hparams["divide_csv_train"] and hparams['data_name'] == 'slurp':
        return vs_train_data, invs_train_data, valid_data, test_data, tokenizer
    else:
        return train_data, valid_data, test_data, tokenizer


def dataio_prepare_nlu(hparams, train_csv, dev_csv, test_csv):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # if split_ratio == None:
    #     print('the split_ratio setting is wrong')
    #     raise ValueError
    role_prefix = ""
    # if data_role == 'aux':
    #     role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]


    assert role_prefix == ""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv,
        # replacements={"data_root": data_folder},
    )





    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=dev_csv,
        # replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv,
        # replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")


    datasets = [train_data, valid_data, test_data]

    # tokenizer = hparams[role_prefix+"tokenizer"]
    if hparams['use_nlu_bert_enc']:
        asr_tokenizer = BertTokenizer.from_pretrained(hparams['slu_enc'].model_name)
    else:
        asr_tokenizer = hparams["asr_tokenizer"] # the "hparams" is actually "nlu_hparams" in the code using dataio_prepare_nlu
    slu_tokenizer = hparams["slu_tokenizer"]


    # # 2. Define input pipeline:
    # @sb.utils.data_pipeline.takes("transcript")
    # @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    # def transcript_pipeline(transcript):
    #     transcript = transcript.upper()
    #     yield transcript
    #     transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
    #     transcript_tokens = torch.LongTensor(transcript_tokens_list)
    #     # print("transcript_tokens is in shape of ", transcript_tokens.shape)
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


    # 3. Define output pipeline:
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

    sb.dataio.dataset.add_dynamic_item(datasets, semantics_pipeline)

    # # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets,
    #     [
    #         "id",
    #         "transcript",
    #         "transcript_tokens",
    #         "semantics",
    #         "semantics_tokens_bos",
    #         "semantics_tokens_eos",
    #         "semantics_tokens",
    #     ],
    # )

    # 4. Set output:
    if hparams['use_nlu_bert_enc']:
        sb.dataio.dataset.set_output_keys(
            datasets,
            [
                "id",
                "transcript",
                "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids",
                "semantics",
                "semantics_tokens_bos",
                "semantics_tokens_eos",
                "semantics_tokens",
            ],
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets,
            [
                "id",
                "transcript",
                "transcript_tokens",
                "semantics",
                "semantics_tokens_bos",
                "semantics_tokens_eos",
                "semantics_tokens",

            ],
        )

    return train_data, valid_data, test_data, asr_tokenizer, slu_tokenizer