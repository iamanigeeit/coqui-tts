import time
import re
import os
import requests
from glob import glob
from pathlib import Path
import traceback
import io
import jiwer
import pandas as pd
from unidecode import unidecode
from TTS.tts.utils.text.cleaners import english_cleaners
from google.cloud import speech
import base64
import json
from vocoder_eval import eval_rmse_f0
from TTS.tts.utils.text import text_to_phones
from TTS.tts.utils.text.symbols import _vowels

VOWEL_REGEX = re.compile(f'[{_vowels.replace(" ", "")}]+')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/perry/PycharmProjects/google_speech_key.json"
VOICEGAIN_JWT_FILE = '/home/perry/PycharmProjects/voicegain_jwt.txt'
with open(VOICEGAIN_JWT_FILE) as f:
    jwt = f.read().strip()
    VOICEGAIN_HEADERS = {'Content-Type': 'application/json',
                         'Accept': 'application/json',
                         'Authorization': f'Bearer {jwt}',
                         }
VOICEGAIN_URL = 'https://api.voicegain.ai/v1/asr/transcribe'


SPLIT_DIR = os.path.realpath(os.path.join(__file__, '../../LJSpeech-1.1/splits'))
SPLIT_CSVS = [os.path.join(SPLIT_DIR, f'{x}.csv') for x in ('train', 'val', 'test')]

GROUND_TRUTH_DIR = os.path.realpath(os.path.join(__file__, '../../LJSpeech-1.1/wavs'))

AUDIO_DETAILS = {
    'encoding': speech.RecognitionConfig.AudioEncoding.LINEAR16,
    'sample_rate_hertz': 22050,
    'language_code': "en-US",
}


def print_all_chars(split_csvs):
    charset = set()
    for split_csv in split_csvs:
        for chars in pd.read_csv(split_csv).raw_text:
            charset.update(set(chars))
    print(''.sorted(charset))

PUNCS = re.compile(r'[!"\'(),-.:;?\[\]’“”]')

def normalize(text):
    return english_cleaners(unidecode(PUNCS.sub(' ', text)))


def google_asr(speech_file):
    """Transcribe the given audio file."""

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(**AUDIO_DETAILS)
    response = client.recognize(config=config, audio=audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    return ' '.join(res.alternatives[0].transcript for res in response.results)


def voicegain_asr(speech_file):
    with open(speech_file, "rb") as f:
        raw_data_str = base64.b64encode(f.read()).decode('ascii')
    data = {"audio": {"source": {"inline": raw_data_str}}}
    response = requests.post(url=VOICEGAIN_URL, headers=VOICEGAIN_HEADERS, data=json.dumps(data))
    if response.ok:
        return response.json()['result']['alternatives'][0]['utterance']
    else:
        raise requests.HTTPError(response.text)


def compute_wer(wav_dir, split_csv, verbose=False):
    true_data = pd.read_csv(split_csv, usecols=['ground_truth_path', 'raw_text'])
    wav_name_to_text = {Path(p).stem: t for p, t in zip(true_data.ground_truth_path, true_data.raw_text)}
    results_csv = os.path.join(wav_dir, 'wer.csv')
    wav_files = [os.path.join(wav_dir, f'{wav_name}.wav') for wav_name in wav_name_to_text]
    transcripts_csv = os.path.join(wav_dir, 'transcripts.csv')
    transcripts = pd.read_csv(transcripts_csv, index_col='wav_name').squeeze()
    assert wav_files, 'No wav files found!'

    results = []
    for wav_file in wav_files:
        wav_name = Path(wav_file).stem
        true_text = wav_name_to_text[wav_name]
        true_text_norm = normalize(true_text)
        transcript_norm = transcripts.loc[wav_file]
        wer = jiwer.wer(truth=true_text_norm, hypothesis=transcript_norm)
        words = len(true_text_norm.split())
        wer_weighted = wer * words
        results.append([wav_name, wer, words, wer_weighted])
        if verbose:
            print(wav_name, f'True text: {true_text_norm}', f'Transcript: {transcript_norm}', wer, sep='\n')
        else:
            print(wav_name, wer)
    results_df = pd.DataFrame(data=results, columns=['wav_name', 'wer', 'words', 'wer_weighted'])
    results_df.set_index('wav_name', inplace=True)
    results_df.sort_index(inplace=True)
    avg_wer = results_df.wer.mean()
    total_words = results_df.words.sum()
    avg_wer_weighted = results_df.wer_weighted.sum() / total_words
    results_df.loc['Total'] = avg_wer, total_words, avg_wer_weighted
    results_df.to_csv(results_csv)
    print(f'Results saved in {results_csv}')


def compute_f0_rmse(wav_dir, ground_truth_dir=GROUND_TRUTH_DIR, verbose=False):
    results_csv = os.path.join(wav_dir, 'f0.csv')
    wav_files = glob(os.path.join(wav_dir, '*.wav'))
    assert wav_files, 'No wav files found!'

    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        results = results_df.values.tolist()[:-1]
        existing_files = set(results_df.wav_name)
        wav_files = [f for f in wav_files if Path(f).stem not in existing_files]
    else:
        results = []
    try:
        for wav_file in wav_files:
            wav_name = Path(wav_file).stem
            gt_wav_file = os.path.join(ground_truth_dir, wav_name + '.wav')

            f0_rmse_mean, vuv_accuracy, vuv_precision = eval_rmse_f0(gt_wav_file, wav_file)
            results.append([wav_name, f0_rmse_mean, vuv_accuracy, vuv_precision])
            if verbose:
                print(wav_name,
                      f'F0 RMSE: {f0_rmse_mean}',
                      f'Accuracy: {vuv_accuracy}',
                      f'Precision: {vuv_precision}', sep='\n')
            else:
                print(wav_name, f0_rmse_mean)
    except:
        traceback.print_exc()
    finally:
        results_df = pd.DataFrame(data=results, columns=['wav_name', 'f0_rmse_mean', 'vuv_accuracy', 'vuv_precision'])
        results_df.set_index('wav_name', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.loc['Total'] = [results_df.f0_rmse_mean.mean(),
                                   results_df.vuv_accuracy.mean(),
                                   results_df.vuv_precision.mean()]
        results_df.to_csv(results_csv)
        print(f'Results saved in {results_csv}')


def transcribe_dir(wav_dir, asr_func):

    results_csv = os.path.join(wav_dir, 'transcripts.csv')
    wav_files = glob(os.path.join(wav_dir, '*.wav'))
    assert wav_files, 'No wav files found!'

    start_time = time.time()
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        results = results_df.values.tolist()[:-1]
        existing_files = set(results_df.wav_name)
        wav_files = [f for f in wav_files if Path(f).stem not in existing_files]
        print(f'Existing: {len(existing_files)}, Remaining: {len(wav_files)}')
    else:
        results = []
    try:
        for wav_file in wav_files:
            wav_name = Path(wav_file).stem
            transcript = asr_func(wav_file)
            transcript_norm = normalize(transcript)
            results.append([wav_name, transcript_norm])
            print(wav_name, transcript_norm)
    except:
        traceback.print_exc()
    finally:
        results_df = pd.DataFrame(data=results, columns=['wav_name', 'transcript'])
        results_df.set_index('wav_name', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.to_csv(results_csv)
        print(f'Results saved in {results_csv}')
    total_time = int(time.time() - start_time)
    time_min = total_time // 60
    time_sec = total_time % 60
    print(f'Took {time_min}:{time_sec}')


def count_syllables(sentence, language='en-us'):
    ipa = text_to_phones(sentence, language=language)
    splits = VOWEL_REGEX.split(ipa)
    return len(splits) - 1


def compute_duration_base(wav_data_csv, test_csv, duration_csv,
                          sample_rate=AUDIO_DETAILS['sample_rate_hertz']):
    wav_data = pd.read_csv(wav_data_csv, index_col='filename')
    test_df = pd.read_csv(test_csv)
    df = test_df[['raw_text', 'ground_truth_path']]
    df['wav_name'] = df.ground_truth_path.apply(lambda x: Path(x).stem)
    df.set_index('wav_name', inplace=True)
    df.sort_index(inplace=True)
    df['duration'] = wav_data.gt_len
    df['duration'] /= sample_rate
    df['syllables'] = df.raw_text.apply(count_syllables)
    df = df[['raw_text', 'syllables', 'duration']]
    total_syllables = df.syllables.sum()
    total_duration = df.duration.sum()
    syllable_speed = str(total_syllables / total_duration)
    df.loc['Total'] = [syllable_speed, total_syllables, total_duration]
    df.to_csv(duration_csv)


def compute_duration(wav_dir, transcript_csv='', stats_csv='',
                     sample_rate=AUDIO_DETAILS['sample_rate_hertz']):
    results_csv = os.path.join(wav_dir, 'duration.csv')
    if not transcript_csv:
        transcript_csv = os.path.join(wav_dir, 'transcripts.csv')
    df = pd.read_csv(transcript_csv, index_col='wav_name')
    df['syllables'] = df.transcript.apply(count_syllables)
    if not stats_csv:
        stats_csv = os.path.join(wav_dir, 'stats.csv')
    stats_df = pd.read_csv(stats_csv, index_col='filename')
    df['duration'] = stats_df.wav_len / sample_rate
    total_syllables = df.syllables.sum()
    total_duration = df.duration.sum()
    syllable_speed = str(total_syllables / total_duration)
    df.loc['Total'] = [syllable_speed, total_syllables, total_duration]
    df.to_csv(results_csv)
    print(f'Results saved in {results_csv}')


def aggregrate_data(in_dirs, filename, out_dir=os.path.dirname(__file__)):
    assert out_dir not in in_dirs
    data = []
    for in_dir in in_dirs:
        full_path = os.path.join(in_dir, filename)
        with open(full_path) as f:
            lines = f.readlines()
            last_line = lines[-1]
            data.append([full_path] + last_line.split(',')[1:])
    columns = ['full_path'] + lines[0].split(',')[1:]
    df = pd.DataFrame(data=data, columns=columns)
    df.set_index('full_path', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(out_dir, filename))


if __name__ == '__main__':
    base_dir = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask'
    wav_dirs = [
        f'{base_dir}/baseline/full_wav_100000_voc0',
        f'{base_dir}/ump/sparsity_20_stop/full_wav_100000_voc0',
        f'{base_dir}/ump/sparsity_40_stop/full_wav_100000_voc0',
        f'{base_dir}/parp/sparsity_20_s0.5_stop/full_wav_100000_voc0',
        f'{base_dir}/parp/sparsity_40_s0.5_stop/full_wav_100000_voc0',
        f'{base_dir}/snip/sparsity_20_stop_batch/full_wav_best_voc0',
        f'{base_dir}/snip/sparsity_40_stop_batch/full_wav_best_voc0',
        f'{base_dir}/sm/sparsity_20_stop_p0.2/full_wav_100000_voc0',
        f'{base_dir}/sm/sparsity_40_stop_p0.2/full_wav_100000_voc0',
        f'{base_dir}/trim/sparsity_2/full_wav_100000_voc0',
        f'{base_dir}/trim/sparsity_4/full_wav_100000_voc0'
    ]

    for wav_dir in wav_dirs:
        print(wav_dir)
        # compute_wer(wav_dir=wav_dir,
        #             split_csv='/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/splits/test.csv',
        #             asr_func=voicegain_asr,
        #             verbose=True)
        # compute_f0_rmse(wav_dir, verbose=True)
        # transcribe_dir(wav_dir, asr_func=voicegain_asr)
        compute_duration(wav_dir.replace('full_', ''))

    # wav_dir = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask/baseline/wav_100000_voc0/'
    # wav_dir = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/wavs/'
    # compute_wer(wav_dir=wav_dir,
    #             split_csv='/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/splits/test.csv',
    #             asr_func=voicegain_asr,
    #             verbose=True)
