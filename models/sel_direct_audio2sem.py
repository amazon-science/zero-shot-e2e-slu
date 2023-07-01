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

from .direct_audio2sem import SLU as direct_SLU

import time
import logging


# Define training procedure

logger = logging.getLogger(__name__)

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

class SLU(direct_SLU):
    def compute_forward(self, batch, nlu=None, stage=sb.Stage.TRAIN, show_results_every=100):
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
        # print(wav2vec2_out.shape)

        # print(stage)
        # print(sb.Stage.TEST)
        # print(str(stage) != str(sb.Stage.TEST))
        # assert str(stage) == str(sb.Stage.TEST)

        if str(stage) != str(sb.Stage.TEST):
        ### for NLU & transcript embedding
            transcript_fea = []
            for sub_id in batch.id:
                transcript_fea.append(nlu[str(sub_id)])
            transcript_fea = torch.cat(transcript_fea, dim=0).cuda()


            audio_fea = wav2vec2_out[:, 0, :]

            ### Cross-Model SelectiveNet forward pass
            sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score\
                = self.modules.cm_sel_net(audio_fea, transcript_fea)


        # SLU forward pass
        e_in = self.hparams.output_emb(tokens_bos)  # torch.Size([4, 147, 128]) # should be the ground truth embedding
        h, _ = self.hparams.dec(e_in, wav2vec2_out, wav_lens) # h: torch.Size([4, 147, 512]) # _ is attention

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)   # logits: torch.Size([4, 147, 58])
        p_seq = self.hparams.log_softmax(logits) # torch.Size([4, 147, 58]) p_seq:


        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            # and self.batch_count % show_results_every != 0
        ):
            return p_seq, wav_lens, sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score
        else:
            # time1 = time.time()
            p_tokens, scores = self.hparams.beam_searcher(
                # wav2vec2_out, wav_lens
                wav2vec2_out.detach(), wav_lens # because the network is large, if no detach, the beam_searcher will dereviate and be very slow
            )
            # time2 = time.time()
            # print('beam_search uses, ', (time2-time1))
            if str(stage) != str(sb.Stage.TEST):
                return p_seq, wav_lens, p_tokens, sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score
            else:
                return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage, show_results_every=100):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            # and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens, sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score\
                = predictions
        else:
            if str(stage) != str(sb.Stage.TEST):
                p_seq, wav_lens, predicted_tokens, sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score \
                = predictions
            else:
                p_seq, wav_lens, p_tokens = predictions


        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ) # NLL loss, not CTC loss

        if str(stage) == str(sb.Stage.TEST):
            loss = loss_seq.mean()
            print('test process, loss is', loss)
        else:
            # add select_loss
            loss_sel_cm = (torch.nn.MSELoss(reduction='mean')(sel_prj_audio_fea, sel_prj_text_fea) * sel_score.view(-1)).mean()
            # loss_aux_cm = (torch.nn.MSELoss(reduction='mean')(aux_prj_audio_fea, aux_prj_text_fea)).mean()
            loss_aux_cm = (torch.nn.MSELoss(reduction='mean')(aux_prj_audio_fea, aux_prj_text_fea) * sel_score.view(-1)).mean()

            emprical_coverage = sel_score.mean()

            # compute emprical risk (=r^)

            loss_sel_cm_div = loss_sel_cm / (emprical_coverage + 0.01)

            # compute penulty (=psi)
            expect_coverage = torch.tensor([self.hparams.selnet_sel_coverage], dtype=torch.float32, requires_grad=True, device='cuda')
            loss_sel_penulty = torch.max(expect_coverage - emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda')) ** 2


            selective_loss = loss_sel_cm_div * self.hparams.selnet_sel_cm_weight + \
                loss_aux_cm * self.hparams.selnet_aux_cm_weight + loss_sel_penulty.sum() * self.hparams.selnet_sel_penality_weight

            loss = selective_loss + (loss_seq * sel_score.view(-1)).mean() / (emprical_coverage + 0.01) * self.hparams.selnet_sel_cm_weight + loss_seq.mean()


        if (stage != sb.Stage.TRAIN):


            predicted_semantics = []
            for utt_seq in predicted_tokens:
                try:
                    predicted_semantics.append(self.tokenizer.decode_ids(utt_seq).split(" "))
                except:
                    predicted_semantics.append(self.tokenizer.decode_ids([0]).split(" "))


            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            # self.log_outputs(predicted_semantics, target_semantics) # can be commented

            if stage != sb.Stage.TRAIN:
                # time1 = time.time()
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                # time2 = time.time()
                # print('!= sb.Stage.TRAIN_1: ', (time2-time1))

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

        # loss = loss_seq
        loss = loss_seq.mean()

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
                # self.hparams["output_folder"] + "/gb_predictions.jsonl", mode="a"
            ) as writer:
                for i in range(len(predicted_semantics)):

                    _dict = {}
                    _dict['entities'] = " ".join(predicted_semantics[i]).replace("|", ",")
                    _dict["ID"] = ids[i]
                    writer.write(_dict)

        return loss





    def fit_batch(self, batch, nlu):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, nlu, sb.Stage.TRAIN)
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

    def evaluate_batch(self, batch, nlu, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, nlu, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:
            # time1 = time.time()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            # time2 = time.time()
            # print('on_stage_start_valid: ', (time2-time1))

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        # print("stage is: ", stage)
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
        # if str(stage) == str(sb.Stage.TRAIN):
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # time1 = time.time()
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
            # time2 = time.time()
            # print('valid_time 1:', (time2-time1))
        # elif stage == sb.Stage.TEST:
        elif str(stage) == "Stage.TEST":
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        nlu=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        if nlu==None:
            raise ValueError('nlu model is not correct set')
        # for params in nlu.parameters():
        #     if params.require_grad != False:
        #         raise ValueError('nlu model is not frozen')

        self.on_fit_start() # resueme a nearest ckpt; file to the model # set optimizer

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    if self._optimizer_step_limit_exceeded:
                        logger.info("Train iteration limit exceeded")
                        break
                    self.step += 1
                    loss = self.fit_batch(batch, nlu)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)   #?

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, nlu, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break