import torch
import speechbrain as sb
import os


def s1_asu_dataio_prepare(hparams, target_name1, target_name2, target_name3):
    """This function prepares the datasets to be used in S1 process, direct train audio->semantics with groundtruth"""

    # data_folder = hparams["data_folder"]



    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name1,
        # replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name2,
        # replacements={"data_root": data_folder},
    )
    try:
        valid_data = valid_data.filtered_sorted(sort_key="duration")
    except:
        pass

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name3,
        # replacements={"data_root": data_folder},
    )
    try:
        test_data = test_data.filtered_sorted(sort_key="duration")
    except:
        pass

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    # def audio_pipeline(wav):
    #     sig = sb.dataio.dataio.read_audio(wav)
    #     return sig
    def audio_pipeline(wav):
        if hparams['data_name'] == "slurp":
            path = hparams['slurp_data_folder'] + '/' + wav
        elif hparams['data_name'] == "slue-voxpopuli":
            file_end = '.ogg'
            # path = hparams['slue_voxpoluli_folder'] +  + key_folder + '/' + wav + file_end
            path = hparams['slue_voxpoluli_folder'] + '/' + wav + file_end
        sig = sb.dataio.dataio.read_audio(path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer




def s2_asu_dataio_prepare(hparams, target_name1, target_name2, target_name3):
    """This function prepares the datasets to be used in S1 process, direct train audio->semantics with groundtruth"""

    # data_folder = hparams["data_folder"]



    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name1,
        # replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name2,
        # replacements={"data_root": data_folder},
    )
    try:
        valid_data = valid_data.filtered_sorted(sort_key="duration")
    except:
        pass

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=target_name3,
        # replacements={"data_root": data_folder},
    )
    try:
        test_data = test_data.filtered_sorted(sort_key="duration")
    except:
        pass

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("tts_wav", "wav")
    @sb.utils.data_pipeline.provides("sig")
    # def audio_pipeline(wav):
    #     sig = sb.dataio.dataio.read_audio(wav)
    #     return sig
    def audio_pipeline(tts_wav, wav):
        path = hparams['slurp_data_folder'] + '/polly_slurp_dataset/' + tts_wav
        flag = False
        if os.path.exists(path):
            path_list = os.listdir(path)
            for ele in path_list:
                if "neural_default_" in ele:
                    path = os.path.join(path, ele)
                    break
            path_list = os.listdir(path)
            for ele in path_list:
                if ".wav" in ele:
                    path = os.path.join(path, ele)
                    flag = True
                    break
            if flag == False:
                print('polly_slurp_dataset misses wav in: ', path)
                path = hparams['slurp_data_folder'] + '/' + wav
        else:
            raise ValueError('the tts_wav folder is missed')
            path = hparams['slurp_data_folder'] + '/' + tts_wav
        # print(path)
        sig = sb.dataio.dataio.read_audio(path)
        return sig


    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer