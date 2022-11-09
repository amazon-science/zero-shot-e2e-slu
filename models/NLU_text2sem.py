
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from enum import Enum, auto
from tqdm.contrib import tqdm

import os

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

# Define training procedure
class NLU(sb.Brain):
    def compute_forward(self, batch, stage, show_results_every=100):
        """Computations from input to semantic outputs"""
        batch = batch.to(self.device)
        # transcript_tokens, transcript_tokens_lens = batch.transcript_tokens
        (
            semantics_tokens_bos,
            semantics_tokens_bos_lens,
        ) = batch.semantics_tokens_bos

        if  self.hparams.use_nlu_bert_enc == True:
            transcript_tokens_input_ids, transcript_tokens_lens = batch.transcript_tokens_input_ids
            transcript_tokens_attention_mask, _ = batch.transcript_tokens_attention_mask
            transcript_tokens_token_type_ids, _ = batch.transcript_tokens_token_type_ids
            # "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"
            transcript_tokens_dict = {
                'input_ids': transcript_tokens_input_ids,
                'attention_mask': transcript_tokens_attention_mask,
                'token_type_ids': transcript_tokens_token_type_ids
            }
            encoder_out = self.hparams.slu_enc(transcript_tokens_dict)
            # print('encoder_out in bert setting has shape of ', encoder_out)
        else:
            transcript_tokens, transcript_tokens_lens = batch.transcript_tokens
            embedded_transcripts = self.hparams.input_emb(transcript_tokens)
            encoder_out = self.hparams.slu_enc(embedded_transcripts)


        e_in = self.hparams.output_emb(semantics_tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, transcript_tokens_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)


        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq, transcript_tokens_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(
                encoder_out, transcript_tokens_lens
            )
            return p_seq, transcript_tokens_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage, show_results_every=100):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, transcript_tokens_lens = predictions
        else:
            p_seq, transcript_tokens_lens, predicted_tokens = predictions

        ids = batch.id
        (
            semantics_tokens_eos,
            semantics_tokens_eos_lens,
        ) = batch.semantics_tokens_eos
        semantics_tokens, semantics_tokens_lens = batch.semantics_tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, semantics_tokens_eos, length=semantics_tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                self.slu_tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            # self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                with jsonlines.open(
                    hparams["output_folder"] + "/predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        dict["file"] = id_to_file[ids[i]]
                        writer.write(dict)

        return loss

    def log_outputs(self, predicted_semantics, target_semantics):
        """ TODO: log these to a file instead of stdout """
        for i in range(len(target_semantics)):
            print(" ".join(predicted_semantics[i]).replace("|", ","))
            print(" ".join(target_semantics[i]).replace("|", ","))
            print("")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:

            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    # def set_hparams(self, main_hparams, nlu_hparams=None):
    #     self.hparams = main_hparams
    #     self.hparams = nlu_hparams

    def infer_forward(self, batch):
        """Computations from input to semantic outputs"""
        batch = batch.to(self.device)
        # transcript_tokens, transcript_tokens_lens = batch.transcript_tokens

        if  self.hparams.use_nlu_bert_enc == True:
            transcript_tokens_input_ids, transcript_tokens_lens = batch.transcript_tokens_input_ids
            transcript_tokens_attention_mask, _ = batch.transcript_tokens_attention_mask
            transcript_tokens_token_type_ids, _ = batch.transcript_tokens_token_type_ids
            # "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"
            transcript_tokens_dict = {
                'input_ids': transcript_tokens_input_ids,
                'token_type_ids': transcript_tokens_token_type_ids,
                'attention_mask': transcript_tokens_attention_mask
            }
            encoder_out = self.hparams.slu_enc(transcript_tokens_dict)
            # print('encoder_out in bert setting has shape of ', encoder_out)
        else:
            transcript_tokens, transcript_tokens_lens = batch.transcript_tokens
            embedded_transcripts = self.hparams.input_emb(transcript_tokens)
            encoder_out = self.hparams.slu_enc(embedded_transcripts)


        p_tokens, scores = self.hparams.beam_searcher(
            encoder_out, transcript_tokens_lens
        )
        return transcript_tokens_lens, p_tokens

    def infer_objectives(self, predictions, batch, slu_tokenizer, save_path, save_file, id_to_file=None):
        """Computes the loss (NLL) given predictions and targets."""
        transcript_tokens_lens, predicted_tokens = predictions

        ids = batch.id

        predicted_semantics = [
            slu_tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in predicted_tokens
        ]

        # write to "predictions.jsonl"
        with jsonlines.open(
            os.path.join(save_path, save_file), mode="a"
        ) as writer:
            for i in range(len(predicted_semantics)):
                try:
                    # dict = ast.literal_eval(
                    #     " ".join(predicted_semantics[i]).replace(
                    #         "|", ","
                    #     )
                    # )
                    dict = ast.literal_eval(" ".join(predicted_semantics[i]).replace("|", ","))
                except SyntaxError:  # need this if the output is not a valid dictionary
                    dict = {
                        "scenario": "none",
                        "action": "none",
                        "entities": [],
                    }

                # below might not be applicable to slue
                if not (type(dict).__name__ == 'dict'):
                    dict = {
                        "scenario": "none",
                        "action": "none",
                        "entities": [],
                    }
                # above might not be applicable to slue


                if id_to_file == None:
                    dict["ID"] = ids[i]
                else:
                    # print(list(id_to_file.keys())[0:10])
                    # print(ids[i])
                    # print(id_to_file[str(ids[i])])
                    dict["file"] = id_to_file[str(ids[i])]
                writer.write(dict)

    def debug_infer_objectives(self, predictions, batch, slu_tokenizer, save_path, save_file, id_to_file=None):
        """Computes the loss (NLL) given predictions and targets."""
        transcript_tokens_lens, predicted_tokens = predictions

        ids = batch.id

        predicted_semantics = [
            slu_tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in predicted_tokens
        ]

        # write to "predictions.jsonl"
        with jsonlines.open(
            os.path.join(save_path, save_file), mode="a"
        ) as writer:
            for i in range(len(predicted_semantics)):
                # try:
                    # dict = ast.literal_eval(
                    #     " ".join(predicted_semantics[i]).replace(
                    #         "|", ","
                    #     )
                    # )
                dict = {}
                dict['entities'] = " ".join(predicted_semantics[i]).replace("|", ",")
                # except SyntaxError:  # need this if the output is not a valid dictionary
                #     dict = {
                #         "scenario": "none",
                #         "action": "none",
                #         "entities": [],
                #     }
                if id_to_file == None:
                    dict["ID"] = ids[i]
                else:
                    dict["file"] = id_to_file[ids[i]]
                writer.write(dict)

    def infer_batch(self, batch, slu_tokenizer, save_path, save_file, id_to_file=None):
        """Computations needed for validation/test batches"""
        predictions = self.infer_forward(batch)
        if 'slurp' in save_path:
            self.infer_objectives(predictions, batch, slu_tokenizer, save_path, save_file, id_to_file)
        elif 'slue-voxpopuli' in save_path:
            self.debug_infer_objectives(predictions, batch, slu_tokenizer, save_path, save_file, id_to_file)


    def infer_syn_label(
        self,
        test_set,
        slu_tokenizer,
        save_path,
        save_file,
        id_to_file=None,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        # self.on_stage_start(Stage.TEST, epoch=None)
        self.batch_count = 0
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                self.infer_batch(batch, slu_tokenizer, save_path, save_file, id_to_file)


                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            # run_on_main(
            #     self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            # )
        self.step = 0
        # return avg_test_loss

    def infer_save_text_fea(self, data_set_list):
        res = {}
        for data_set in data_set_list:
            with tqdm(data_set, initial=0, dynamic_ncols=True, disable=False) as t:
                for batch in t:
                    id_str = str(batch['id'])


                    if self.hparams.use_nlu_bert_enc == True:
                        transcript_tokens_input_ids = batch['transcript_tokens_input_ids'].unsqueeze(0).cuda()
                        transcript_tokens_attention_mask = batch['transcript_tokens_attention_mask'].unsqueeze(0).cuda()
                        transcript_tokens_token_type_ids = batch['transcript_tokens_token_type_ids'].unsqueeze(0).cuda()
                        # "transcript_tokens_input_ids", "transcript_tokens_attention_mask", "transcript_tokens_token_type_ids"
                        transcript_tokens_dict = {
                            'input_ids': transcript_tokens_input_ids,
                            'attention_mask': transcript_tokens_attention_mask,
                            'token_type_ids': transcript_tokens_token_type_ids
                        }
                        transcript_encoder_out = self.hparams.slu_enc(transcript_tokens_dict)
                        # print('encoder_out in bert setting has shape of ', transcript_encoder_out)
                    else:
                        transcript_tokens = batch['transcript_tokens']
                        transcript_tokens = transcript_tokens.unsqueeze(0).cuda()

                        embedded_transcripts = self.hparams.input_emb(transcript_tokens)
                        transcript_encoder_out = self.hparams.slu_enc(embedded_transcripts)

                    # ### for NLU & transcript embedding
                    # transcript_tokens = batch['transcript_tokens']
                    # transcript_tokens = transcript_tokens.unsqueeze(0).cuda()
                    #
                    # embedded_transcripts = self.hparams.input_emb(transcript_tokens)
                    # transcript_encoder_out = self.hparams.slu_enc(embedded_transcripts)  # called as slu_enc, but it actually is the nlu enc



                    # need add condition to judge whether -1 or 0
                    transcript_fea = transcript_encoder_out[:, -1, :]  # it is LSTM, so it is -1
                    res[id_str] = transcript_fea.detach().cpu()
        return res
