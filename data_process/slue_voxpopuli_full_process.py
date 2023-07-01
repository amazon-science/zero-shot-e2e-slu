import speechbrain as sb


def dataio_prepare(hparams, data_role):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    role_prefix = ""
    if data_role == 'aux':
        role_prefix = "aux_"
    data_folder = hparams[role_prefix+"data_folder"]

    if hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli-full':
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

    if hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli-full':
        datasets = [vs_train_data, invs_train_data, valid_data, test_data]
    else:
        datasets = [train_data, valid_data, test_data]

    tokenizer = hparams[role_prefix+"tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        key_folder = wav[0:4]
        file_end = '.ogg'
        path = hparams['slue_voxpoluli_full_folder'] + '/' + key_folder + '/' + wav + file_end

        sig = sb.dataio.dataio.read_audio(path)
        return sig

    # sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) # cal audio signal

    # 2. Define id pipeline:
    @sb.utils.data_pipeline.takes("ID")
    @sb.utils.data_pipeline.provides("ID")
    def id_pipeline(ID):
        return ID



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
        ["id", "transcript"],
        # ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens", "transcript"],
    )
    if hparams["divide_csv_train"] and hparams['data_name'] == 'slue-voxpopuli-full':
        return vs_train_data, invs_train_data, valid_data, test_data, tokenizer
    else:
        return train_data, valid_data, test_data, tokenizer