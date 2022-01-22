import collections
import os
import random
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.tts.utils.text import pad_with_eos_bos
from TTS.utils.audio import AudioProcessor


"""
[data|train|val]_items refers to the list of [text, wav_path, speaker, ...] returned by `load_train_eval_items`.
samples refers to the list of dicts passed to batching functions which includes symbol_ids, phoneme_ids, mels, etc.
"""

class TTSDataset(Dataset):
    def __init__(
        self,
        dataset_config: Dict,
        ap: AudioProcessor,
        items: List[List],
        outputs_per_step: int = 1,
        use_symbols: bool = False,
        use_phonemes: bool = True,
        use_mel: bool = True,
        use_linear: bool = False,
        use_f0: bool = False,
        use_wav: bool = False,
        batch_group_size: int = 0,
        min_seq_len: int = 0,
        max_seq_len: int = 2147483647,  # max int32
        enable_eos_bos: bool = False,
        use_noise_augment: bool = False,
        speaker_id_mapping: Dict = None,
        d_vector_mapping: Dict = None,
        verbose: bool = True,
    ):
        """Generic 📂 data loader for `tts` models. It is configurable for different outputs and needs.

        If you need something different, you can inherit and override.

        Args:
            outputs_per_step (int): Number of time frames predicted per step.

            text_cleaner (list): List of text cleaners to clean the input text before converting to sequence IDs.

            compute_linear_spec (bool): compute linear spectrogram if True.

            ap (TTS.tts.utils.AudioProcessor): Audio processor object.

            meta_data (list): List of dataset instances.

            compute_f0 (bool): compute f0 if True. Defaults to False.

            f0_cache_path (str): Path to store f0 cache. Defaults to None.

            character_config (dict): `dict` of custom text characters used for converting texts to sequences.

            custom_symbols (list): List of custom symbols used for converting texts to sequences. Models using its own
                set of symbols need to pass it here. Defaults to `None`.

            add_blank (bool): Add a special `blank` character after every other character. It helps some
                models achieve better results. Defaults to false.

            return_wav (bool): Return the waveform of the sample. Defaults to False.

            batch_group_size (int): Range of batch randomization after sorting
                sequences by length. It shuffles each batch with bucketing to gather similar lenght sequences in a
                batch. Set 0 to disable. Defaults to 0.

            min_seq_len (int): Minimum input sequence length to be processed
                by sort_inputs`. Filter out input sequences that are shorter than this. Some models have a
                minimum input length due to its architecture. Defaults to 0.

            max_seq_len (int): Maximum input sequence length. Filter out input sequences that are longer than this.
                It helps for controlling the VRAM usage against long input sequences. Especially models with
                RNN layers are sensitive to input length. Defaults to `Inf`.

            use_phonemes (bool): If true, input text converted to phonemes. Defaults to false.

            phoneme_cache_path (str): Path to cache computed phonemes. It writes phonemes of each sample to a
                separate file. Defaults to None.

            phoneme_language (str): One the languages from supported by the phonemizer interface. Defaults to `en-us`.

            enable_eos_bos (bool): Enable the `end of sentence` and the `beginning of sentences characters`. Defaults
                to False.

            speaker_id_mapping (dict): Mapping of speaker names to IDs used to compute embedding vectors by the
                embedding layer. Defaults to None.

            d_vector_mapping (dict): Mapping of wav files to computed d-vectors. Defaults to None.

            use_noise_augment (bool): Enable adding random noise to wav for augmentation. Defaults to False.

            verbose (bool): Print diagnostic information. Defaults to false.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.ap = ap
        self.items = items
        self.outputs_per_step = outputs_per_step
        self.use_symbols = use_symbols
        self.use_phonemes = use_phonemes
        self.use_mel = use_mel
        self.use_linear = use_linear
        self.use_wav = use_wav
        self.use_f0 = use_f0
        self.batch_group_size = batch_group_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.enable_eos_bos = enable_eos_bos
        self.use_noise_augment = use_noise_augment
        self.speaker_id_mapping = speaker_id_mapping
        self.d_vector_mapping = d_vector_mapping
        self.verbose = verbose

        assert not (use_symbols and use_phonemes), "Can't use both symbols and phonemes"
        if verbose:
            print("\n > DataLoader initialization")
            local_vars = locals()
            inputs_used = []
            for inp in ["symbols", "phonemes", "mel", "linear", "wav", "f0"]:
                if local_vars["use_" + inp]:
                    inputs_used.append(inp)
            print(" | > Inputs used: {}".format(', '.join(inputs_used)))
            print(" | > Number of instances : {}".format(len(self.items)))

        self.char_ids_all = None
        self.id_lengths = None
        char_ids_cache_path = None
        if self.use_symbols:
            char_ids_cache_path = self.dataset_config["symbol_ids_cache_path"]
        elif self.use_phonemes:
            char_ids_cache_path = self.dataset_config["phoneme_ids_cache_path"]
        if char_ids_cache_path:
            self.char_ids_all = [np.load(get_npy_path(char_ids_cache_path, item[1])) for item in self.items]
            self.id_lengths = np.array([len(char_ids) for char_ids in self.char_ids_all], dtype=np.int32)

        # TODO: stop hardcoding?
        self.eos_id = 1
        self.bos_id = 2
        
        if use_f0:
            self.pitch_extractor = PitchExtractor(items)
            self.f0_cache_path = dataset_config["f0_cache_path"]
            self.pitch_extractor.load_pitch_stats(self.f0_cache_path)

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    def load_data(self, idx):
        """
        From data_item, load samples. Do not add processing at this stage, do it in collate_fn. 
        Args:
            idx:

        Returns:

        """
        item = self.items[idx]

        if len(item) == 4:
            text, wav_path, speaker_name, attn_file = item
            attn = np.load(attn_file)
        else:
            text, wav_path, speaker_name = item
            attn = None

        char_ids = None
        char_length = None
        if self.char_ids_all:
            char_ids = self.char_ids_all[idx]
            char_length = self.id_lengths[idx]
            assert char_length > 0, f" [!] Input sequence is empty for {wav_path}"

            if len(char_ids) > self.max_seq_len:
                # return a different sample if the phonemized
                # text is longer than the threshold
                # TODO: find a better fix
                return self.load_data(1)

        mel = None
        if self.use_mel:
            mel_path = get_npy_path(self.dataset_config["mel_cache_path"], wav_path)
            mel = np.load(mel_path)

        linear = None
        if self.use_linear:
            linear_path = get_npy_path(self.dataset_config["linear_cache_path"], wav_path)
            linear = np.load(linear_path)

        wav = None
        if self.use_wav:
            wav = np.asarray(self.load_wav(wav_path), dtype=np.float32)
            assert wav.size > 0, f" [!] wav is empty for {self.items[idx][1]}"
            # apply noise for augmentation
            if self.use_noise_augment:
                wav = wav + (1.0 / 32768.0) * np.random.rand(*wav.shape)

        pitch = None
        if self.use_f0:
            pitch = self.pitch_extractor.load_pitch(self.f0_cache_path, wav_path)
            pitch = self.pitch_extractor.normalize_pitch(pitch)

        sample = {
            "text": text,
            "char_ids": char_ids,
            "char_length": char_length,
            "mel": mel,
            "linear": linear,
            "wav": wav,
            "pitch": pitch,
            "attn": attn,
            "speaker_name": speaker_name,
            "wav_path": wav_path,
        }
        return sample

    def sort_and_filter_items(self, by_audio_len=False):
        r"""Sort `items` based on text length or audio length in ascending order. Filter out samples out or the length
        range.

        Args:
            by_audio_len (bool): if True, sort by audio length else by text length.
        """
        # compute the target sequence length
        if by_audio_len:
            lengths = []
            for item in self.items:
                lengths.append(os.path.getsize(item[1]) / 16 * 8)  # assuming 16bit audio
                lengths = np.array(lengths, dtype=np.int32)
        else:
            lengths = self.id_lengths

        # sort items based on the sequence length in ascending order
        idxs = np.argsort(lengths)
        new_items = [self.items[idx] for idx in idxs]
        new_char_ids = [self.char_ids_all[idx] for idx in idxs]
        lengths = lengths[idxs]
        start = np.searchsorted(lengths, self.min_seq_len)
        stop = np.searchsorted(lengths, self.max_seq_len, side='right')
        new_items = new_items[start:stop]
        new_char_ids = new_char_ids[start:stop]
        lengths = lengths[start:stop]
        dropped_count = len(self.items) - len(new_items)

        # shuffle batch groups
        # create batches with similar length items
        # the larger the `batch_group_size`, the higher the length variety in a batch.
        # TODO: what's the point of this? new_items is already sorted, we don't need to shuffle chunks
        if self.batch_group_size > 0:
            for i in range(len(new_items) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                shuffle_idxs = list(range(offset, end_offset))
                random.shuffle(shuffle_idxs)
                new_items[offset:end_offset] = [new_items[idx] for idx in shuffle_idxs]
                if self.char_ids_all:
                    new_char_ids[offset:end_offset] = [new_char_ids[idx] for idx in shuffle_idxs]
                    lengths[offset:end_offset] = [lengths[idx] for idx in shuffle_idxs]

        if len(new_items) == 0:
            raise RuntimeError(" [!] No items left after filtering.")

        # update items to the new sorted items
        self.items = new_items
        self.char_ids_all = new_char_ids
        self.id_lengths = lengths

        # logging
        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(
                " | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                    self.max_seq_len, self.min_seq_len, dropped_count)
                )
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    @staticmethod
    def _sort_batch(batch, id_lengths):
        """Sort the batch by the input text length for RNN efficiency.

        Args:
            batch (Dict): Batch returned by `__getitem__`.
            id_lengths (List[int]): Lengths of the input character sequences.
        """
        id_lengths, ids_sorted_decreasing = torch.sort(torch.IntTensor(id_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, id_lengths, ids_sorted_decreasing

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to features.
        3. PAD sequences wrt r.
        4. Load to Torch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):

            id_lengths = [len(sample["char_ids"]) for sample in batch]

            # sort items with text input length for RNN efficiency
            batch, id_lengths, ids_sorted_decreasing = self._sort_batch(batch, id_lengths)

            # convert list of dicts to dict of lists
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}

            # get pre-computed d-vectors
            if self.d_vector_mapping is not None:
                wav_files = [batch["wav_path"][idx] for idx in ids_sorted_decreasing]
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in wav_files]
            else:
                d_vectors = None

            # get numerical speaker ids from speaker names
            if self.speaker_id_mapping:
                speaker_ids = [self.speaker_id_mapping[sn] for sn in batch["speaker_name"]]
            else:
                speaker_ids = None

            # compute features
            mel = batch["mel"]

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # compute 'stop token' targets
            stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            char_ids = batch["char_ids"]
            if self.enable_eos_bos:
                char_ids = [[self.bos_id] + sample_char_ids + [self.eos_id] for sample_char_ids in char_ids]
            char_ids = prepare_data(char_ids).astype(np.int32)

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            id_lengths = torch.IntTensor(id_lengths)
            char_ids = torch.IntTensor(char_ids)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.IntTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)

            if speaker_ids is not None:
                speaker_ids = torch.IntTensor(speaker_ids)

            # compute linear spectrogram
            if self.use_linear:
                linear = [self.ap.spectrogram(w).astype("float32") for w in batch["wav"]]
                linear = prepare_tensor(linear, self.outputs_per_step)
                linear = linear.transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()
            else:
                linear = None

            # format waveforms
            wav_padded = None
            if self.use_wav:
                # wav_lengths = [w.shape[0] for w in batch["wav"]]
                max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                # wav_lengths = torch.IntTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch["wav"]), 1, max_wav_len)
                for i, w in enumerate(batch["wav"]):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
                wav_padded.transpose_(1, 2)

            # compute f0
            # TODO: compare perf in collate_fn vs in load_data
            if self.use_f0:
                pitch = prepare_data(batch["pitch"])
                assert mel.shape[1] == pitch.shape[1], f"[!] {mel.shape} vs {pitch.shape}"
                pitch = torch.FloatTensor(pitch)[:, None, :].contiguous()  # B x 1 xT
            else:
                pitch = None

            # collate attention alignments
            if batch["attn"][0]:
                attns = [batch["attn"][idx].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = char_ids.shape[1] - attn.shape[0]
                    assert pad1 >= 0 and pad2 >= 0, f"[!] Negative padding - {pad1} and {pad2}"
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)
            else:
                attns = None
            # TODO: return dictionary
            return {
                "char_ids": char_ids,
                "id_lengths": id_lengths,
                "speaker_names": batch["speaker_name"],
                "linear": linear,
                "mel": mel,
                "mel_lengths": mel_lengths,
                "stop_targets": stop_targets,
                "wav_path": batch["wav_path"],
                "d_vectors": d_vectors,
                "speaker_ids": speaker_ids,
                "attns": attns,
                "waveform": wav_padded,
                "text": batch["text"],
                "pitch": pitch,
            }

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )


class PitchExtractor:
    """Pitch Extractor for computing F0 from wav files.

    Args:
        items (List[List]): Dataset samples.
        verbose (bool): Whether to print the progress.
    """


    def __init__(
            self,
            items: List[List],
    ):
        self.items = items
        self.mean = None
        self.std = None


    @staticmethod
    def _compute_and_save_pitch(ap, wav_file, pitch_file=None):
        wav = ap.load_wav(wav_file)
        pitch = ap.compute_f0(wav)
        if pitch_file:
            np.save(pitch_file, pitch)
        return pitch


    @staticmethod
    def compute_pitch_stats(pitch_vecs):
        nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in pitch_vecs])
        mean, std = np.mean(nonzeros), np.std(nonzeros)
        return mean, std


    def normalize_pitch(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch = pitch - self.mean
        pitch = pitch / self.std
        pitch[zero_idxs] = 0.0
        return pitch


    def denormalize_pitch(self, pitch):
        zero_idxs = np.where(pitch == 0.0)[0]
        pitch *= self.std
        pitch += self.mean
        pitch[zero_idxs] = 0.0
        return pitch


    @staticmethod
    def load_pitch(cache_path, wav_file):
        """
        compute pitch and return a numpy array of pitch values
        """
        pitch_file = get_npy_path(cache_path, wav_file)
        pitch = np.load(pitch_file, allow_pickle=True)
        return pitch.astype(np.float32)


    @staticmethod
    def _pitch_worker(args):
        item, ap, cache_path = args
        _, wav_file, *_ = item
        pitch_file = get_npy_path(cache_path, wav_file)
        if not os.path.exists(pitch_file):
            pitch = PitchExtractor._compute_and_save_pitch(ap, wav_file, pitch_file)
            return pitch
        return None


    def compute_pitch(self, ap, cache_path, num_workers=0):
        """Compute the input sequences with multi-processing.
        Call it before passing dataset to the data loader to cache the input sequences for faster data loading."""

        print(" | > Computing pitch features ...")
        if num_workers == 0:
            pitch_vecs = []
            for _, item in enumerate(tqdm(self.items)):
                pitch_vecs.append(self._pitch_worker([item, ap, cache_path]))
        else:
            with Pool(num_workers) as p:
                pitch_vecs = list(
                    tqdm(
                        p.imap(PitchExtractor._pitch_worker, [[item, ap, cache_path] for item in self.items]),
                        total=len(self.items),
                    )
                )
        pitch_mean, pitch_std = self.compute_pitch_stats(pitch_vecs)
        pitch_stats = {"mean": pitch_mean, "std": pitch_std}
        np.save(os.path.join(cache_path, "pitch_stats"), pitch_stats, allow_pickle=True)


    def load_pitch_stats(self, cache_path):
        stats_path = os.path.join(cache_path, "pitch_stats.npy")
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)


def get_npy_path(cache_path, wav_path):
    return os.path.join(cache_path, Path(wav_path).stem + ".npy")