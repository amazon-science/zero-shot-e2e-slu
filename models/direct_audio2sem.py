#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU.
We encode input waveforms into features using the wav2vec2/HuBert model,
then feed the features into a seq2seq model to map them to semantics.
(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)
Run using:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Boumadane Abdelmoumene 2021
 * AbdelWahab Heba 2021
 * Yingzhi Wang 2021
For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf
"""

import speechbrain as sb

import jsonlines
import ast
import torch


from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from enum import Enum, auto
from tqdm.contrib import tqdm
from speechbrain.utils.distributed import run_on_main
import os

import time

# Define training procedure

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

class SLU(sb.Brain):
    def compute_forward(self, batch, stage, show_results_every=100):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        #  encoder forward pass
        wav2vec2_out = self.modules.wav2vec2(wavs)  # torch.Size([4, 201, 768])

        # SLU forward pass
        e_in = self.hparams.output_emb(tokens_bos)  # torch.Size([4, 147, 128]) # should be the ground truth embedding
        h, _ = self.hparams.dec(e_in, wav2vec2_out, wav_lens) # h: torch.Size([4, 147, 512]) # _ is attention

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)   # logits: torch.Size([4, 147, 58])
        p_seq = self.hparams.log_softmax(logits) # torch.Size([4, 147, 58]) p_seq:


        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq, wav_lens
        else:
            # time1 = time.time()
            p_tokens, scores = self.hparams.beam_searcher(
                wav2vec2_out, wav_lens
            )
            # time2 = time.time()
            # print('beam_search uses, ', (time2-time1))

            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage, show_results_every=100):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ) # NLL loss, not CTC loss

        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):

            predicted_semantics = []
            for utt_seq in predicted_tokens:
                try:
                    predicted_semantics.append(self.tokenizer.decode_ids(utt_seq).split(" "))
                except:
                    predicted_semantics.append(self.tokenizer.decode_ids([0]).split(" "))


            target_semantics = [wrd.split(" ") for wrd in batch.semantics]


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
                    os.path.join(save_path, save_name), mode="a"
                    # self.hparams["output_folder"] + "/gb_predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            _dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                            if not isinstance(_dict, dict):
                                _dict = {
                                    "scenario": "none",
                                    "action": "none",
                                    "entities": [],
                                }
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            _dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        _dict["file"] = id_to_file[ids[i]]
                        writer.write(_dict)

        return loss

    def infer_objectives(self, predictions, batch, stage, save_path, save_name, id_to_file=None, show_results_every=100):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ) # NLL loss, not CTC loss

        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):


            # below is self-write
            predicted_semantics = []
            for utt_seq in predicted_tokens:
                try:
                    predicted_semantics.append(self.tokenizer.decode_ids(utt_seq).split(" "))
                except:
                    predicted_semantics.append(self.tokenizer.decode_ids([0]).split(" "))

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            # self.log_outputs(predicted_semantics, target_semantics) # can be commented

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )


            with jsonlines.open(
                os.path.join(save_path, save_name), mode="a"
            ) as writer:
                for i in range(len(predicted_semantics)):
                    try:
                        _dict = ast.literal_eval(
                            " ".join(predicted_semantics[i]).replace(
                                "|", ","
                            )
                        )
                        if not isinstance(_dict, dict):
                            _dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                    except:  # need this if the output is not a valid dictionary
                        _dict = {
                            "scenario": "none",
                            "action": "none",
                            "entities": [],
                        }
                    _dict["file"] = id_to_file[ids[i]]
                    writer.write(_dict)

        return loss

    def debug_infer_objectives(self, predictions, batch, stage, save_path, save_name, id_to_file=None, show_results_every=100):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ) # NLL loss, not CTC loss

        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):

            predicted_semantics = []
            for utt_seq in predicted_tokens:
                try:
                    predicted_semantics.append(self.tokenizer.decode_ids(utt_seq).split(" "))
                except:
                    predicted_semantics.append(self.tokenizer.decode_ids([0]).split(" "))

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            # self.log_outputs(predicted_semantics, target_semantics) # can be commented

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

            # if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
            with jsonlines.open(
                os.path.join(save_path, save_name), mode="a"
                # self.hparams["output_folder"] + "/gb_predictions.jsonl", mode="a"
            ) as writer:
                for i in range(len(predicted_semantics)):
                    # try:
                        # _dict = ast.literal_eval(
                        #     " ".join(predicted_semantics[i]).replace(
                        #         "|", ","
                        #     )
                        # )
                    _dict = {}
                    _dict['entities'] = " ".join(predicted_semantics[i]).replace("|", ",")
                    _dict["ID"] = ids[i]
                    writer.write(_dict)

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
            self.wav2vec2_optimizer.step()
            self.optimizer.step()
        self.wav2vec2_optimizer.zero_grad()   # two optimizer, needs check
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()


    def infer_batch(self, batch, stage, save_path, save_name, id_to_file):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        if 'slurp' in save_path:
            loss = self.infer_objectives(predictions, batch, stage, save_path, save_name, id_to_file)
        elif 'slue-voxpopuli' in save_path:
            loss = self.debug_infer_objectives(predictions, batch, stage, save_path, save_name, id_to_file)

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
        # print("stage is: ", stage)
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        # elif stage == sb.Stage.TEST:
        elif str(stage) == "Stage.TEST":
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"   # initialize the optimizer with Adam optimizer for wav2vec2
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters()) # initialize the optimizer with Adam optimizer for other parts in the modules

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)


    def infer_objectives_loss(
        self,
        test_set,
        save_path,
        save_name,
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
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss = self.infer_batch(batch, stage=Stage.TEST, save_path=save_path, save_name=save_name, id_to_file=id_to_file)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0
        return avg_test_loss