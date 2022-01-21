import sys
import math
import gruut
import pandas as pd
from typing import List, Tuple, Union

from TTS.tts.datasets.dataset import *
from TTS.tts.datasets.formatters import *

"""
Update: i wanted to be able to adjust the train/val/test split ratios and save the splits for reproducibility.
Once the splits are saved, there is no need to use meta_file_train and meta_file_val in `shared_configs.py`
    and we can simply use `meta_file` (especially since only ljspeech formats make it easy to split meta_file
    into train and val).
"""

# This is a hack because Gruut phonemizer cannot be pickled and thus cannot be passed to `phoneme_worker`.
# However, creating a new phonemizer for each worker is very slow.
phonemizers = []


def set_phonemizers(phoneme_language, phonemizer_args, use_espeak_phonemes, num_workers):
    if use_espeak_phonemes:
        # Use a lexicon/g2p model train on eSpeak IPA instead of gruut IPA.
        # This is intended for backwards compatibility with TTS<=v0.0.13
        # pre-trained models.
        phonemizer_args["model_prefix"] = "espeak"
    global phonemizers
    phonemizers = []
    if phonemizer_args:
        for i in range(num_workers):
            phonemizers.append(gruut.get_phonemizer(phoneme_language, **phonemizer_args))
    else:
        for i in range(num_workers):
            phonemizers.append(gruut.get_phonemizer(phoneme_language))
    return phoneme_language


def split_dataset(items, split_ratios):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.

    Args:
        items (List[List]): A list of samples. Each sample is a list of `[text, audio_path, speaker_name]`.
        split_ratios (Tuple[float]): (train_ratio, val_ratio, test_ratio)
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    assert train_ratio + val_ratio + test_ratio == 1, ' [!] Sum of train/val/test ratios should be 1'

    items_df = pd.DataFrame(items, columns=('input_seq_path', 'ground_truth_path', 'speaker_name'))
    assert train_ratio * len(items_df) > 100, " [!] You do not have enough samples to train. You need at least 100 samples."

    def get_split_sizes(items):
        total_items = len(items)
        val_size = math.ceil(val_ratio * total_items)
        test_size = math.ceil(test_ratio * total_items)
        train_size = total_items - val_size - test_size
        return train_size, val_size, test_size

    train_dfs, val_dfs, test_dfs = [], [], []
    for speaker in set(items_df.speaker_name):
        speaker_items = items_df[items_df.speaker_name == speaker]
        indices = np.random.permutation(speaker_items.index)
        train_size, val_size, test_size = get_split_sizes(speaker_items)
        train_dfs.append(speaker_items.loc[indices[:train_size]])
        val_dfs.append(speaker_items.loc[indices[train_size:-test_size]])
        test_dfs.append(speaker_items.loc[indices[-test_size:]])
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    return train_df, val_df, test_df


def load_train_eval_items(
    dataset_configs: Union[List[Dict], Dict],
    dataset_items: Union[List[List[List]], List[List]] = None,
) -> Tuple[List[List], List[List]]:
    """Load data samples as a List of `[[text, audio_path, speaker_name], ...]]` and load attention alignments if provided.
    If `dataset_items` is provided, split each dataset into samples saved to `[train|val|test]_df_file`.
    Otherwise, directly load train and eval samples from `train_df_file` and `val_df_file`.
    Returns train and eval samples combined across datasets.

    Args:
        dataset_configs (List[Dict], Dict): A list of datasets or a single dataset dictionary. If multiple datasets are
            in the list, they are merged.

        dataset_items (List[List[List]], optional): A list of `items` = `[[text, audio_path, speaker_name], ...]]`
            usually returned by `preprocess()`. Defaults to None (load from train_df_file.

    Returns:
        Tuple[List[List], List[List]: training and evaluation splits of combined datasets.
    """

    train_items_all = []
    val_items_all = []
    if not isinstance(dataset_configs, list):
        dataset_configs = [dataset_configs]
        dataset_items = [dataset_items]
    for dataset_config, items in zip(dataset_configs, dataset_items):
        train_df_file = dataset_config["train_df_file"]
        val_df_file = dataset_config["val_df_file"]
        if items is None:
            assert train_df_file and val_df_file, " [!] train_df_file and val_df_file must be given if no dataset_items"
            train_df = pd.read_csv(train_df_file)
            val_df = pd.read_csv(val_df_file)
        else:
            split_ratios = dataset_config["split_ratios"]
            assert len(split_ratios) == 3, " [!] split_ratios must be (train_ratio, val_ratio, test_ratio)"
            train_df, val_df, test_df = split_dataset(items, dataset_config['split_ratios'])
            split_dir = os.path.dirname(train_df_file)
            os.makedirs(split_dir, exist_ok=True)
            train_df.to_csv(train_df_file, index=False)
            val_df.to_csv(val_df_file, index=False)
            test_df.to_csv(os.path.join(split_dir, 'test.csv'), index=False)
            print(f" | > Wrote splits to {split_dir}: "
                  f"{len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")

        # load attention masks for the duration predictor training
        meta_file_attn_mask = dataset_config["meta_file_attn_mask"]
        if meta_file_attn_mask:
            audio_to_attn = dict(load_attention_mask_meta_data(meta_file_attn_mask))
            train_df['attn_path'] = train_df.audio_path.map(audio_to_attn)
            val_df['attn_path'] = val_df.audio_path.map(audio_to_attn)

        train_items_all.extend(train_df.values.tolist())
        val_items_all.extend(val_df.values.tolist())
    print(f" | > Train/eval splits loaded.")
    return train_items_all, val_items_all


def load_attention_mask_meta_data(metafile_path: str) -> List[Tuple]:
    """Load meta data file created by compute_attention_masks.py"""
    with open(metafile_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta_data = []
    for line in lines:
        wav_file, attn_file = line.split("|")
        meta_data.append((wav_file, attn_file))
    return meta_data


def _get_formatter_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def preprocess(dataset_config: Dict, ap: AudioProcessor, num_workers: int = 0):
        
    name = dataset_config["name"]
    root_path = dataset_config["root_path"]
    meta_file = dataset_config["meta_file_train"]
    formatter = _get_formatter_by_name(name)
    items = formatter(root_path, meta_file)
    num_items = len(items)
    print(f" Preprocessing {name}...")
    print(f" | > Found {num_items}) files in {Path(root_path).resolve()}")
    ap = ap

    # TODO: need to ensure wav filenames are unique.
    mel_cache_path = dataset_config["mel_cache_path"]
    if mel_cache_path:
        cache_existed = make_cache_dir(mel_cache_path, num_items)
        if not cache_existed:
            compute_spectrogram(mel_cache_path, num_workers, mel=True)

    linear_cache_path = dataset_config["linear_cache_path"]
    if linear_cache_path:
        cache_existed = make_cache_dir(linear_cache_path, num_items)
        if not cache_existed:
            compute_spectrogram(linear_cache_path, num_workers, mel=False)

    symbol_ids_cache_path = dataset_config["symbol_ids_cache_path"]
    cleaner_name = dataset_config["cleaner_name"]
    character_config = dataset_config["character_config"]
    custom_symbols = dataset_config["custom_symbols"]
    add_blank = dataset_config["add_blank"]
    if symbol_ids_cache_path:
        cache_existed = make_cache_dir(symbol_ids_cache_path, num_items)
        if not cache_existed:
            compute_symbol_ids(items, symbol_ids_cache_path, cleaner_name,
                               custom_symbols, character_config, add_blank, num_workers)

    phoneme_ids_cache_path = dataset_config["phoneme_ids_cache_path"]
    phoneme_language = dataset_config["phoneme_language"]
    phonemizer_args = dataset_config["phonemizer_args"]
    use_espeak_phonemes = dataset_config["use_espeak_phonemes"]
    if phoneme_ids_cache_path:
        cache_existed = make_cache_dir(phoneme_ids_cache_path, num_items)
        if not cache_existed:
            set_phonemizers(phoneme_language, phonemizer_args, use_espeak_phonemes, num_workers)
            compute_phoneme_ids(items, phoneme_ids_cache_path, cleaner_name, phoneme_language,
                                custom_symbols, character_config, add_blank, num_workers)

    f0_cache_path = dataset_config["f0_cache_path"]
    if f0_cache_path:
        cache_existed = make_cache_dir(f0_cache_path, num_items)
        if not cache_existed:
            pitch_extractor = PitchExtractor(items)
            pitch_extractor.compute_pitch(ap, f0_cache_path, num_workers)

    return items
    

def make_cache_dir(cache_path, num_items):
    cache_exists = os.path.isdir(cache_path)
    if cache_exists:
        file_count = len(next(os.walk(cache_path))[2])
        if file_count < num_items:
            cache_exists = False
    else:
        os.makedirs(cache_path, exist_ok=True)
    return cache_exists


def _spectrogram_worker(args):
    item, load_wav, spec_fn, cache_path = args
    wav_path = item[1]
    spectrogram_path = get_npy_path(cache_path, wav_path)
    if not os.path.exists(spectrogram_path):
        audio = load_wav(wav_path)
        spectrogram = spec_fn(audio).astype("float32")
        np.save(spectrogram_path, spectrogram)


def compute_spectrogram(items, ap, cache_path, num_workers=0, mel=True):
    """Compute the input sequences with multi-processing."""
    if mel:
        spec_fn = ap.melspectrogram
        spec_type = "mels"
    else:
        spec_fn = ap.spectrogram
        spec_type = "linear spectrograms"
    print(f" | > Computing {spec_type} ...")
    if num_workers == 0:
        for item in tqdm(items):
            _spectrogram_worker([item, ap.load_wav, spec_fn, cache_path])
    else:
        with Pool(num_workers) as p:
            list(tqdm(
                p.imap(
                    _spectrogram_worker,
                    [[item, ap.load_wav, spec_fn, cache_path] for item in items]),
                total=len(items),
                )
            )


def _symbol_worker(args):
    item, cache_path, cleaner_name, custom_symbols, character_config, add_blank = args
    text, wav_path, *_ = item
    symbol_ids_path = get_npy_path(cache_path, wav_path)
    if not os.path.exists(symbol_ids_path):
        symbol_ids = text_to_symbol_ids(
            text,
            cleaner_name,
            custom_symbols=custom_symbols,
            character_config=character_config,
            add_blank=add_blank,
        )
        symbol_ids = np.asarray(symbol_ids, dtype=np.int32)
        np.save(symbol_ids_path, symbol_ids)
    return


def compute_symbol_ids(items, cache_path, cleaner_name,
                       custom_symbols, character_config, add_blank, num_workers):
    print(" | > Computing symbol ID sequences...")
    if num_workers == 0:
        for item in tqdm(items):
            _symbol_worker((item, cache_path, cleaner_name, custom_symbols, character_config, add_blank))
    else:
        with Pool(num_workers) as p:
            list(tqdm(
                p.imap(_symbol_worker,
                       [(item, cache_path, cleaner_name, custom_symbols, character_config, add_blank)
                        for item in items]),
                total=len(items),
            ))


def _phoneme_worker(args):
    item, cache_path, cleaner_name, phoneme_language, worker_idx, custom_symbols, character_config, add_blank = args
    text, wav_path, *_ = item
    phoneme_ids_path = get_npy_path(cache_path, wav_path)
    if not os.path.exists(phoneme_ids_path):
        phonemizer = phonemizers[worker_idx]
        phoneme_ids = text_to_phoneme_ids(
            text,
            cleaner_name=cleaner_name,
            language=phoneme_language,
            phonemizer=phonemizer,
            custom_symbols=custom_symbols,
            character_config=character_config,
            add_blank=add_blank,
        )
        phoneme_ids = np.asarray(phoneme_ids, dtype=np.int32)
        np.save(phoneme_ids_path, phoneme_ids)


def compute_phoneme_ids(items, cache_path, cleaner_name, phoneme_language,
                        custom_symbols, character_config, add_blank, num_workers):
    print(" | > Computing phoneme ID sequences...")
    if num_workers == 0:
        for item in tqdm(items):
            _phoneme_worker((item, cache_path, cleaner_name, phoneme_language, 0,
                            custom_symbols, character_config, add_blank))
    else:
        with Pool(num_workers) as p:
            list(tqdm(
                p.imap(_phoneme_worker,
                       [(item, cache_path, cleaner_name, phoneme_language, i % num_workers,
                         custom_symbols, character_config, add_blank) for i, item in enumerate(items)]),
                total=len(items),
            ))




