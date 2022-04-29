import json
import os
import random
import re
from psytoolkit_templates import *

import shutil

MOS_TEST_FILES = ['LJ041-0031.wav', 'LJ030-0095.wav', 'LJ003-0347.wav', 'LJ035-0036.wav', 'LJ004-0210.wav',
                  'LJ029-0053.wav', 'LJ037-0247.wav', 'LJ048-0216.wav', 'LJ019-0097.wav', 'LJ050-0050.wav']

PREF3_TEST_FILES = ['LJ049-0093.wav', 'LJ042-0107.wav', 'LJ037-0084.wav', 'LJ033-0039.wav', 'LJ029-0029.wav',
                   'LJ024-0107.wav', 'LJ019-0383.wav', 'LJ016-0145.wav', 'LJ007-0052.wav', 'LJ003-0192.wav']

TTS_DIR = '/home/perry/PycharmProjects/TTS'
NATURAL_DIR = f'{TTS_DIR}/recipes/ljspeech/LJSpeech-1.1/wavs'
BASE_DIR = f'{TTS_DIR}/recipes/ljspeech/prune/tacotron2_nomask'

WEBAPP_DIR = f'{TTS_DIR}/webapp/public'
HOSTING_URL = 'https://hosting-cfcdd.web.app'
MOS_DIRNAME = 'mos_test_files'

MOS_WAVDIR_PREFIX = {NATURAL_DIR: 'natural',
                     f'{BASE_DIR}/baseline/wav_100000_voc0/': 'baseline',
                     f'{BASE_DIR}/parp/sparsity_20_s0.5_stop/wav_100000_voc0/': 'parp20',
                     f'{BASE_DIR}/parp/sparsity_40_s0.5_stop/wav_100000_voc0/': 'parp40',
                     f'{BASE_DIR}/snip/sparsity_20_stop_batch/wav_best_voc0/': 'snip20',
                     f'{BASE_DIR}/snip/sparsity_40_stop_batch/wav_best_voc0/': 'snip40',
                     f'{BASE_DIR}/sm/sparsity_20_stop_p0.2/wav_100000_voc0/': 'sm20',
                     f'{BASE_DIR}/sm/sparsity_40_stop_p0.2/wav_100000_voc0/': 'sm40',
                     f'{BASE_DIR}/trim/sparsity_2/wav_100000_voc0/': 'trim2',
                     f'{BASE_DIR}/trim/sparsity_4/wav_100000_voc0/': 'trim4', }


PREF3_GROUPS = ['snip', 'sm']

SNIP_WAVDIR_PREFIX = {
    f'{BASE_DIR}/baseline/wav_100000_voc0/': 'baseline',
    f'{BASE_DIR}/snip/sparsity_20_stop_batch/wav_best_voc0/': 'snip20',
    f'{BASE_DIR}/snip/sparsity_40_stop_batch/wav_best_voc0/': 'snip40', }

SM_WAVDIR_PREFIX = {
    f'{BASE_DIR}/baseline/wav_100000_voc0/': 'baseline',
    f'{BASE_DIR}/sm/sparsity_20_stop_p0.2/wav_100000_voc0/': 'sm20',
    f'{BASE_DIR}/sm/sparsity_40_stop_p0.2/wav_100000_voc0/': 'sm40', }

PREF3_JSON_PATH = 'psytoolkit_pref3.json'

NONWORDS = re.compile(r'\W')


def format_wav_str(s):
    return NONWORDS.sub('_', s[:-4])


def copy_to_webapp(test_files, wavdir_prefix, output_dir):
    """Copies and renames wav files to host on firebase."""
    if os.path.exists(output_dir):
        assert not os.listdir(output_dir), "Output dir not empty"
    else:
        os.mkdir(output_dir)
    for wavdir, prefix in wavdir_prefix.items():
        for wav_file in test_files:
            shutil.copyfile(os.path.join(wavdir, wav_file),
                            f'{output_dir}/{prefix}_{wav_file}')


def compile_mos(mos_dir, hosting_url=HOSTING_URL):
    dirname = os.path.basename(mos_dir)
    mos_qns = []
    for wav_file in sorted(os.listdir(mos_dir)):
        mos_qn = MOS_QN_TEMPLATE.format(
            wav_file_underscore=format_wav_str(wav_file),
            hosting_url=hosting_url,
            dirname=dirname,
            wav_file=wav_file,
        )
        mos_qns.append(mos_qn)
    mos_questions = ''.join(mos_qns)
    return MOS_SECTION_TEMPLATE.format(mos_questions=mos_questions)


def compile_pref3(pref3_dirs, groups, group_prefixes, test_files, hosting_url=HOSTING_URL, json_path=PREF3_JSON_PATH):
    pref3_qns = []
    qn_wav_order = {}
    for pref3_dir, group, prefixes in zip(pref3_dirs, groups, group_prefixes):
        dirname = os.path.basename(pref3_dir)
        for wav_file in test_files:
            shuffle_prefixes = prefixes.copy()
            random.shuffle(shuffle_prefixes)
            wav_files = [f'{prefix}_{wav_file}' for prefix in shuffle_prefixes]
            pref3_qn = PREF3_QN_TEMPLATE.format(
                group=group,
                wav_file_underscore=format_wav_str(wav_file),
                hosting_url=hosting_url,
                dirname=dirname,
                wav_files=wav_files,
            )
            pref3_qns.append(pref3_qn)
            qn_name = re.search(r'l: (.*)', pref3_qn).groups()[0]
            qn_wav_order[qn_name] = shuffle_prefixes
    pref3_questions = ''.join(pref3_qns)
    with open(json_path, 'w') as f:
        json.dump(qn_wav_order, f)
    return PREF3_TEST_TEMPLATE.format(pref3_questions=pref3_questions)


if __name__ == '__main__':
    mos_dir = os.path.join(WEBAPP_DIR, MOS_DIRNAME)
    copy_to_webapp(test_files=MOS_TEST_FILES, wavdir_prefix=MOS_WAVDIR_PREFIX,
                   output_dir=mos_dir)
    mos_section = compile_mos(mos_dir=mos_dir)
    print(mos_section)

    pref3_dirs = [os.path.join(WEBAPP_DIR, f'pref3_{group}_files') for group in PREF3_GROUPS]
    group_prefixes = [SNIP_WAVDIR_PREFIX.values(), SM_WAVDIR_PREFIX.values()]
    for pref3_dir in pref3_dirs:
        copy_to_webapp(test_files=PREF3_TEST_FILES, wavdir_prefix=SNIP_WAVDIR_PREFIX,
                       output_dir=pref3_dir)

    pref3_section = compile_pref3(pref3_dirs=pref3_dirs,
                                  groups=PREF3_GROUPS,
                                  group_prefixes=group_prefixes,
                                  test_files=PREF3_TEST_FILES,
                                  )
    print(pref3_section)


