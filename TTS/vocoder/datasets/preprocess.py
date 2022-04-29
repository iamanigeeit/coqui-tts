import glob
import os
from pathlib import Path
from multiprocessing import Pool
from typing import List

import numpy as np
from coqpit import Coqpit
from tqdm import tqdm

from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import get_npy_path
from TTS.tts.datasets import make_cache_dir

def preprocess_wav_files(out_path: str, config: Coqpit, ap: AudioProcessor):
    """Process wav and compute mel and quantized wave signal.
    It is mainly used by WaveRNN dataloader.

    Args:
        out_path (str): Parent folder path to save the files.
        config (Coqpit): Model config.
        ap (AudioProcessor): Audio processor.
    """
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    wav_files = find_wav_files(config.data_path)
    for path in tqdm(wav_files):
        wav_name = Path(path).stem
        quant_path = os.path.join(out_path, "quant", wav_name + ".npy")
        mel_path = os.path.join(out_path, "mel", wav_name + ".npy")
        y = ap.load_wav(path)
        mel = ap.melspectrogram(y)
        np.save(mel_path, mel)
        if isinstance(config.mode, int):
            quant = ap.mulaw_encode(y, qc=config.mode) if config.model_args.mulaw else ap.quantize(y, bits=config.mode)
            np.save(quant_path, quant)


def find_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    return wav_paths


def find_feat_files(data_path):
    feat_paths = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
    return feat_paths


def load_wav_data(data_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    np.random.seed(0)
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def load_wav_feat_data(data_path, feat_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    feat_paths = find_feat_files(feat_path)

    wav_paths.sort(key=lambda x: Path(x).stem)
    feat_paths.sort(key=lambda x: Path(x).stem)

    assert len(wav_paths) == len(feat_paths), f" [!] {len(wav_paths)} vs {feat_paths}"
    for wav, feat in zip(wav_paths, feat_paths):
        wav_name = Path(wav).stem
        feat_name = Path(feat).stem
        assert wav_name == feat_name

    items = list(zip(wav_paths, feat_paths))
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]


def compute_normalized_mel(data_dir: str, mel_norm_cache_path: str, vocoder_config: Coqpit, ap: AudioProcessor):
    wav_paths = find_wav_files(data_dir)
    cache_existed = make_cache_dir(mel_norm_cache_path, len(wav_paths))
    if not cache_existed:
        assert vocoder_config.audio.signal_norm
        print(f" | > Computing normalized mel ...")
        num_workers = vocoder_config.num_loader_workers
        if num_workers == 0:
            for wav_path in tqdm(wav_paths):
                _spectrogram_worker([wav_path, ap.load_wav, ap.melspectrogram, mel_norm_cache_path])
        else:
            with Pool(num_workers) as p:
                list(tqdm(
                    p.imap(
                        _spectrogram_worker,
                        [[wav_path, ap.load_wav, ap.melspectrogram, mel_norm_cache_path] for wav_path in wav_paths]),
                    total=len(wav_paths),
                    )
                )


def _spectrogram_worker(args):
    wav_path, load_wav, spec_fn, cache_path = args
    spectrogram_path = get_npy_path(cache_path, wav_path)
    if not os.path.exists(spectrogram_path):
        audio = load_wav(wav_path)
        spectrogram = spec_fn(audio).astype("float32")
        np.save(spectrogram_path, spectrogram)


def load_data_from_splits(csvs, mel_norm_cache_path):
    import pandas as pd
    if isinstance(csvs, str):
        wav_paths = pd.read_csv(csvs).ground_truth_path.values.tolist()
        feats = [get_npy_path(mel_norm_cache_path, wav_path) for wav_path in wav_paths]
        return list(zip(wav_paths, feats))
    else:
        items_list = []
        for csv in csvs:
            wav_paths = pd.read_csv(csv).ground_truth_path.values.tolist()
            feats = [get_npy_path(mel_norm_cache_path, wav_path) for wav_path in wav_paths]
            items = list(zip(wav_paths, feats))
            items_list.append(items)
        return items_list
