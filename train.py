import os
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.decoders.scorer import ScorerBuilder
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import sentencepiece as spm
import sys

from huggingface_hub import login, HfApi,create_repo
from pathlib import Path
import os

# Load token từ biến môi trường
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("⚠️ HF_TOKEN chưa được đặt trong biến môi trường!")

# Đăng nhập
login(token=token)

def upload_checkpoint_to_hf(checkpoint_dir, repo_id, epoch):
    api = HfApi()
    # Tạo repository nếu chưa tồn tại
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    # Duyệt qua tất cả file trong thư mục checkpoint
    for file_path in checkpoint_dir.glob("*"):
        if file_path.is_file():
            path_in_repo = f"checkpoints/epoch_{epoch}/{file_path.name}"
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded {file_path.name} to {path_in_repo}")

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig # (B, N)
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs) # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            # Note: For valid_search, for the sake of efficiency, we only perform beamsearch with
            # limited capacity and no LM to give user some idea of how the AM is doing

            # Decide searcher for inference: valid or test search
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), wav_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), wav_lens
                )

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_lens
                )
                tokens_eos = self.hparams.fea_augment.replicate_labels(
                    tokens_eos
                )
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
           ckpts, recoverable_name="model",
        )
        
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Lưu checkpoint cục bộ
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

            # Tìm thư mục checkpoint mới nhất
            checkpoint_dirs = sorted(
                self.checkpointer.checkpoints_dir.glob("CKPT*"),
                key=lambda x: x.stat().st_mtime,  # Sửa từ mtime thành st_mtime
                reverse=True
            )
            if checkpoint_dirs:
                latest_checkpoint_dir = checkpoint_dirs[0]
                # Tải tất cả file trong thư mục checkpoint lên Hugging Face
                upload_checkpoint_to_hf(latest_checkpoint_dir, "MinhNghia/speechCheckPoint", epoch)

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )
            
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

def dataio_prepare(hparams):
    """Chuẩn bị dataset từ file CSV."""
    train_data = DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": hparams["data_folder"]}
    )
    valid_data = DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": hparams["data_folder"]}
    )
    datasets = [train_data, valid_data]
    tokenizer = spm.SentencePieceProcessor(model_file=hparams["tokenizer_model"])

    @sb.utils.data_pipeline.takes("wav", "wrd")
    @sb.utils.data_pipeline.provides("sig", "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def audio_text_pipeline(wav, wrd):
        audio = sb.dataio.dataio.read_audio(wav)
        yield audio
        yield wrd
        tokens_list = tokenizer.encode(wrd, out_type=int)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, audio_text_pipeline)
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
    )
    return train_data, valid_data, tokenizer, None, None

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # from prepare_vivos import prepare_vivos_hf
    # run_on_main(
    #     prepare_vivos_hf,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["output_folder"],
    #         "splits": hparams["train_splits"] + hparams["dev_splits"] + hparams["test_splits"]
    #     }
    # )

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )
    train_data, valid_data, tokenizer, _, _ = dataio_prepare(hparams)
    with torch.autograd.detect_anomaly():
      asr_brain.fit(
          epoch_counter=hparams["epoch_counter"],
          train_set=train_data,
          valid_set=valid_data,
          train_loader_kwargs=hparams["train_dataloader_opts"],
          valid_loader_kwargs=hparams["valid_dataloader_opts"]
      )
    # for test_set_name, test_set in test_datasets.items():
    #     asr_brain.hparams.output_wer_folder = os.path.join(hparams["output_folder"], test_set_name)
    #     asr_brain.evaluate(
    #         test_set,
    #         min_key="WER",
    #         test_loader_kwargs=hparams["test_dataloader_opts"]
    #     )