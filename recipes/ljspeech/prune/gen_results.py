import glob
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt

from TTS.tts.models import setup_model as setup_tts_model
import torch
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from TTS.tts.datasets.dataset import TTSDataset
from TTS.utils.audio import AudioProcessor
from TTS.config import load_config
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.utils.generic_utils import get_npy_path
from TTS.vocoder.datasets.preprocess import load_data_from_splits
from TTS.tts.utils.synthesis import trim_silence
from TTS.tts.utils.visual import plot_spectrogram
import wave


BASE_DIR = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/'
DATA_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/"
GT_WAV_DIR = DATA_DIR + "wavs/"
GT_STATS_CSV = DATA_DIR + "wav_data.csv"
TEST_CSV = DATA_DIR + "splits/test.csv"
MODEL_DIR = BASE_DIR + "tacotron2_nomask/"
ZERO_DIR = MODEL_DIR + 'baseline/'
CONFIG_PATH = ZERO_DIR + "config.json"
STEPS = '100000'
ZERO_MODEL_FILE = ZERO_DIR + f"checkpoint_{STEPS}.pth.tar"
BATCH_SIZE = 32


VOCODER_BASE_DIR = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune_vocoder/'
VOCODER_DIR = VOCODER_BASE_DIR + 'default_batch_16/'
VOCODER_ZERO_DIR = VOCODER_DIR + 'sparsity_0/'
VOCODER_ZERO_FILE = VOCODER_ZERO_DIR + "checkpoint_200000.pth.tar"

def get_config(sparsity_dir):
    config = load_config(os.path.join(sparsity_dir, 'config.json'))
    config.audio['do_trim_silence'] = False
    ap = AudioProcessor(**config.audio)
    return config, ap


def get_mel_dir(sparsity_dir, steps=STEPS):
    return os.path.join(sparsity_dir, f"mel_{steps}")

def get_full_mel_dir(sparsity_dir, steps=STEPS):
    return os.path.join(sparsity_dir, f"full_mel_{steps}")

def get_wav_gl_dir(sparsity_dir, steps=STEPS):
    return os.path.join(sparsity_dir, f"wav_gl_{steps}")

def get_wav_gan_dir(sparsity_dir, steps=STEPS):
    return os.path.join(sparsity_dir, f"wav_gan_{steps}")

def setup_loader(config, ap, csv_path, exclude_filenames=set(), batch_size=BATCH_SIZE):
    items = pd.read_csv(csv_path).values.tolist()
    if exclude_filenames:
        items = [item for item in items if Path(item[1]).stem not in exclude_filenames]
    dataset = TTSDataset(
        config.dataset_configs[0],
        ap,
        items,
        use_phonemes=config.use_phonemes,
        use_mel=config.use_mel,
        enable_eos_bos=config.enable_eos_bos,
    )
    dataset.sort_and_filter_items()
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False
    )
    return loader


def find_filenames(dirpath):
    return {Path(x).stem for x in os.listdir(dirpath)}


def compute_test_mels(sparsity_dir, config, model_filename, loader, steps=STEPS):
    model_file = os.path.join(sparsity_dir, model_filename)
    model = setup_tts_model(config)
    model.load_checkpoint(config, model_file, eval=True)
    model = model.cuda()
    with torch.no_grad():
        mel_dir = get_mel_dir(sparsity_dir, steps)
        full_mel_dir = get_full_mel_dir(sparsity_dir, steps)
        os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(full_mel_dir, exist_ok=True)
        wav_filenames = []
        output_lengths_all = []
        mel_gt_lengths_all = []
        for batch in loader:
            char_ids = batch["char_ids"]
            wav_path = batch["wav_path"]
            wav_filenames.extend([Path(wav_file).stem for wav_file in wav_path])
            mel_gt_lengths = batch["mel_lengths"]
            mel_gt_lengths_all.extend(mel_gt_lengths.tolist())
            char_ids = char_ids.cuda()

            results = model.inference(char_ids)

            model_outputs = results['model_outputs']
            stop_tokens = results['stop_tokens']

            output_lengths = torch.sum(stop_tokens < 0.5, dim=1).squeeze()
            output_lengths_all.extend(output_lengths.tolist())

            for model_output, output_length, mel_gt_length, wav_file in zip(
                    model_outputs, output_lengths, mel_gt_lengths, wav_path):
                model_output = model_output[:output_length, :]
                spectrogram_full = model_output.cpu().numpy().squeeze().T
                np.save(get_npy_path(full_mel_dir, wav_file), spectrogram_full)
                model_out = model_output[:mel_gt_length, :]
                spectrogram = model_out.cpu().numpy().squeeze().T
                np.save(get_npy_path(mel_dir, wav_file), spectrogram)

        length_data = pd.DataFrame(data=list(zip(mel_gt_lengths_all, output_lengths_all)),
                                   columns=['mel_gt_length', 'output_length'],
                                   index=wav_filenames)

        length_data.to_csv(os.path.join(mel_dir, 'length_data.csv'))



def save_mel_plot(args):
    mel_path, plots_dir = args
    spec = np.load(mel_path)
    fig = plot_spectrogram(spec.T)
    fig.savefig(os.path.join(plots_dir, Path(mel_path).stem + '.png'), bbox_inches='tight')


def save_mel_plots(mel_dir, plots_dir, num_workers=4):
    os.makedirs(plots_dir, exist_ok=True)
    mel_paths = glob.glob(os.path.join(mel_dir, '*.npy'))
    with Pool(num_workers) as p:
        list(tqdm(p.imap(save_mel_plot, [(mel_path, plots_dir) for mel_path in mel_paths]), total=len(mel_paths)))


def compute_test_wavs(mel_dir, wav_out_dir, ttm_ap=None, vocoder_file=''):
    if ttm_ap:
        ap = ttm_ap
        vocoder_config = None
    else:
        assert vocoder_file
        vocoder_sparsity_dir = os.path.dirname(vocoder_file)
        vocoder_config, ap = get_config(vocoder_sparsity_dir)

    mel_paths = glob.glob(os.path.join(mel_dir, '*.npy'))
    wav_stats = []
    stats_path = os.path.join(wav_out_dir, 'stats.csv')
    if os.path.exists(wav_out_dir):
        existing_filenames = find_filenames(wav_out_dir)
        mel_paths = [mel_path for mel_path in mel_paths if Path(mel_path).stem not in existing_filenames]
        if os.path.exists(stats_path):
            wav_stats = pd.read_csv(stats_path).values.tolist()
    else:
        os.makedirs(wav_out_dir)

    gt_stats = get_wav_gt_stats()

    if vocoder_file:
        vocoder = setup_vocoder_model(vocoder_config)
        vocoder.load_checkpoint(vocoder_config, vocoder_file, eval=True)
        vocoder = vocoder.cuda()
        with torch.no_grad():
            for mel_path in mel_paths:
                spectrogram = np.load(mel_path)
                vocoder_input = torch.tensor(spectrogram).unsqueeze(0).cuda()
                waveform = vocoder.inference(vocoder_input)
                waveform = waveform.cpu().numpy().squeeze()
                waveform = ap.trim_silence(waveform)
                filename = Path(mel_path).stem
                ap.save_wav(waveform, os.path.join(wav_out_dir, filename + '.wav'))
                wav_stats.append([filename, gt_stats[filename], len(waveform)])
    else:
        for mel_path in mel_paths:
            spectrogram = np.load(mel_path)
            waveform = ap.inv_melspectrogram(spectrogram)
            waveform = ap.trim_silence(waveform)
            filename = Path(mel_path).stem
            ap.save_wav(waveform, os.path.join(wav_out_dir, filename + '.wav'))
            wav_stats.append([filename, gt_stats[filename], len(waveform)])

    wav_stats_df = pd.DataFrame(data=wav_stats, columns=['filename', 'gt_len', 'wav_len'])
    wav_stats_df.set_index('filename', inplace=True)
    wav_stats_df.sort_index(inplace=True)
    wav_stats_df.to_csv(stats_path)


def get_wav_gt_stats():
    if os.path.exists(GT_STATS_CSV):
        gt_stats_sr = pd.read_csv(GT_STATS_CSV, index_col='filename', squeeze=True)
        gt_stats = gt_stats_sr.to_dict()
    else:
        gt_stats = {}
        gt_paths = glob.glob(os.path.join(GT_WAV_DIR, '*.wav'))
        for gt_path in gt_paths:
            with wave.open(gt_path, 'rb') as w:
                gt_len = w.getnframes()
            gt_stats[Path(gt_path).stem] = gt_len
        gt_stats_sr = pd.Series(data=gt_stats)
        gt_stats_sr.index.name = 'filename'
        gt_stats_sr.name = 'gt_len'
        gt_stats_sr.sort_index(inplace=True)
        gt_stats_sr.to_csv(GT_STATS_CSV)
    return gt_stats


def get_wav_stats(wav_out_dir):
    wav_stats = pd.DataFrame(index=pd.Index(name='filename', data=[]), columns=['gt_len', 'wav_len'])
    gt_stats = get_wav_gt_stats()
    for wav_path in glob.glob(os.path.join(wav_out_dir, '*.wav')):
        filename = Path(wav_path).stem
        with wave.open(wav_path, 'rb') as w:
            wav_len = w.getnframes()
        wav_stats.loc[filename] = (gt_stats[filename], wav_len)
    wav_stats.sort_index(inplace=True)
    wav_stats.to_csv(os.path.join(wav_out_dir, 'stats.csv'))



if __name__ == '__main__':

    # mel_dir = os.path.join(ZERO_DIR, f'mel_{STEPS}')
    # wav_out_dir = os.path.join(ZERO_DIR, f'wav_{STEPS}_voc0')
    #
    # exclude_filenames = find_filenames(mel_dir)
    config, ap = get_config(ZERO_DIR)
    loader = setup_loader(config, ap, TEST_CSV)  #, exclude_filenames=exclude_filenames)
    #
    # compute_test_mels(ZERO_DIR, config, 'checkpoint_100000.pth.tar', loader)
    # compute_test_wavs(mel_dir, wav_out_dir, vocoder_file=VOCODER_ZERO_FILE)

    # get_wav_stats(wav_out_dir)

    base_dir = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask'
    mel_dirs = []
    wav_out_dirs = []
    for x in ['2', '4']:
        folder = f'sparsity_{x}0_stop'
        mel_dirs.append(f'{base_dir}/ump/{folder}/full_mel_100000')
        wav_out_dirs.append(f'{base_dir}/ump/{folder}/full_wav_100000_voc0')
        folder = f'sparsity_{x}0_stop_batch'
        mel_dirs.append(f'{base_dir}/snip/{folder}/full_mel_best')
        wav_out_dirs.append(f'{base_dir}/snip/{folder}/full_wav_best_voc0')
        folder = f'sparsity_{x}0_stop_p0.2'
        mel_dirs.append(f'{base_dir}/sm/{folder}/full_mel_100000')
        wav_out_dirs.append(f'{base_dir}/sm/{folder}/full_wav_100000_voc0')
        folder = f'sparsity_{x}'
        mel_dirs.append(f'{base_dir}/trim/{folder}/full_mel_100000')
        wav_out_dirs.append(f'{base_dir}/trim/{folder}/full_wav_100000_voc0')
    for mel_dir, wav_out_dir in zip(mel_dirs, wav_out_dirs):
        # sparsity_dir = os.path.join(MODEL_DIR, 'trim', f'sparsity_{i}')
        # mel_dir = os.path.join(sparsity_dir, f'full_mel_100000')
        # wav_out_dir = os.path.join(sparsity_dir, f'full_wav_100000_voc0')
        # compute_test_mels(sparsity_dir, config, 'checkpoint_100000.pth.tar', loader)
        compute_test_wavs(mel_dir, wav_out_dir, vocoder_file=VOCODER_ZERO_FILE)
    #
    # get_wav_gt_stats()
    # get_wav_stats(ZERO_DIR + 'full_wav_voc_sparsity_0/')
    #
    # for d in [f'sparsity_{x}_stop_p0.2' for x in [40]]: #[10, 20, 30, 40, 50, 60, 70]]:
    #     sparsity_dir = os.path.join(MODEL_DIR, 'sm', d)
    #     mel_dir = os.path.join(sparsity_dir, f'mel_{STEPS}')
    #     wav_out_dir = os.path.join(sparsity_dir, f'wav_{STEPS}_voc0')
    #     compute_test_mels(sparsity_dir, config, 'checkpoint_100000.pth.tar', loader)
    #     compute_test_wavs(mel_dir, wav_out_dir, vocoder_file=VOCODER_ZERO_FILE)

    # from recipes.ljspeech.prune.gen_results import save_mel_plot
    # dirpath = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask/sm/sparsity_40_s0.5_stop_p0.2_resume/'
    # save_mel_plots(dirpath + 'mel_100000/', dirpath + 'mel_plots_100000/')
