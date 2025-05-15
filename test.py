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
from speechbrain.utils.distributed import run_on_main, if_main_process

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

        # Không cần tính loss trong giai đoạn test
        loss = torch.tensor(0.0).to(self.device)

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                if stage == sb.Stage.TEST:
                    # Lưu ID và transcription cho test
                    for id_, pred in zip(ids, predicted_words):
                        transcription = " ".join(pred)
                        self.predictions.append((id_, transcription))

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint load if needed"""
        super().on_evaluate_start()
        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        if ckpts:
            ckpt = torch.load(ckpts[0], map_location=self.device)  # Load the latest checkpoint
            self.hparams.model.load_state_dict(ckpt["model"], strict=True)
            self.hparams.model.eval()
            print(f"Loaded checkpoint: {ckpts[0]}")
        else:
            print("No checkpoints found, using current model state")

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            if stage == sb.Stage.TEST:
                self.predictions = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TEST:
            if if_main_process():
                wer_dir = os.path.join(hparams["output_folder"], "test")
                os.makedirs(wer_dir, exist_ok=True)
                predictions_file = os.path.join(wer_dir, "predictions.txt")
                with open(predictions_file, "w", encoding="utf-8") as f:
                    f.write("ID\tTranscription\n")
                    for id_, transcription in self.predictions:
                        f.write(f"{id_}\t{transcription}\n")
            
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

def dataio_prepare(hparams):
    test_datasets = {
        Path(csv_file).stem: DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": hparams["data_folder"]}
        ) for csv_file in hparams["test_csv"]
    }
    datasets = list(test_datasets.values())
    tokenizer = spm.SentencePieceProcessor(model_file=hparams["tokenizer_model"])

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "tokens_bos", "tokens_eos", "tokens")
    def audio_pipeline(wav):
        audio = sb.dataio.dataio.read_audio(wav)
        yield audio
        # Vì không có wrd, tạo token giả hoặc bỏ qua nếu không cần
        tokens_list = []  # Hoặc để trống nếu không cần token
        yield torch.LongTensor([hparams["bos_index"]] + tokens_list)  # tokens_bos
        yield torch.LongTensor(tokens_list + [hparams["eos_index"]])  # tokens_eos
        yield torch.LongTensor(tokens_list)  # tokens

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"]
    )
    return test_datasets, tokenizer, None, None

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )
    test_datasets, tokenizer, _, _ = dataio_prepare(hparams)
    
    for test_set_name, test_set in test_datasets.items():
        print("Testing on", test_set_name)
        print("Dataset:", test_set)
        wer_file = os.path.join(hparams["output_folder"], "test", "wer.txt")
        asr_brain.hparams.test_wer_file = wer_file
        wer_dir = os.path.dirname(wer_file)
        if wer_dir:
            print(f"Creating directory: {wer_dir}")
            os.makedirs(wer_dir, exist_ok=True)
        asr_brain.evaluate(
            test_set,
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_opts"]
        )