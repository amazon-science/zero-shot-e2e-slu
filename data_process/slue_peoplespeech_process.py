import torch
import speechbrain as sb


def dataio_prepare(hparams, data_role):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    if hparams[role_prefix+"divide_csv_train"]:
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
    # valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[role_prefix+"csv_test"], replacements={"data_root": data_folder},
    )
    # test_data = test_data.filtered_sorted(sort_key="duration")

    if hparams[role_prefix+"divide_csv_train"]:
        datasets = [vs_train_data, invs_train_data, valid_data, test_data]
    else:
        datasets = [train_data, valid_data, test_data]

    tokenizer = hparams[role_prefix+"tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("ID", "split")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(ID, split):
        path = ID + split + ".ogg"
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    # sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal

    # 2. Define id pipeline:
    @sb.utils.data_pipeline.takes("ID")
    @sb.utils.data_pipeline.provides("ID")
    def id_pipeline(ID):
        return ID

    # sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline) # cal audio signal

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

    # sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define text-transcript pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("text")
    def transcript_pipeline(text):
        yield text


    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "text"],
        # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )
    if hparams[role_prefix+"divide_csv_train"]:
        return vs_train_data, invs_train_data, valid_data, test_data, tokenizer
    else:
        return train_data, valid_data, test_data, tokenizer