import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_train_eval_items, preprocess
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer

output_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.realpath(os.path.join(output_path, "../LJSpeech-1.1/"))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech",
    root_path=data_path,
    meta_file_train=os.path.join(data_path, "metadata.csv"),
    ununsed_speakers=[],
    meta_file_val="",
    meta_file_attn_mask="",
    train_df_file=os.path.join(data_path, 'splits', 'train.csv'),
    val_df_file=os.path.join(data_path, 'splits', 'val.csv'),
    split_ratios=[0.8, 0.1, 0.1],
    cleaner_name="english_cleaners",
    symbol_ids_cache_path=os.path.join(data_path, 'symbol_ids'),
    phoneme_ids_cache_path=os.path.join(data_path, 'phoneme_ids'),
    mel_cache_path=os.path.join(data_path, 'mel'),
    linear_cache_path=os.path.join(data_path, 'linear'),
    f0_cache_path=os.path.join(data_path, 'f0'),
    character_config={},
    custom_symbols=[],
    add_blank=False,
    phoneme_language="en-us",
    phonemizer_args={
        "remove_stress": True,
        "ipa_minor_breaks": False,  # don't replace commas/semi-colons with IPA |
        "ipa_major_breaks": False,  # don't replace periods with IPA ‖
    },
    use_espeak_phonemes=False,
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20.0,
    preemphasis=0.0,
)



# init audio processor
ap = AudioProcessor(**audio_config.__dict__)

# preprocess if necessary
# all_items = preprocess(dataset_config, ap, num_workers=6)
all_items = None

# load training samples
train_items, eval_items = load_train_eval_items(dataset_config, all_items)


config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=32,
    eval_batch_size=32,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    ga_alpha=5.0,
    r=1,
    attention_type="dynamic_convolution",
    double_decoder_consistency=True,
    # epochs=1,
    epochs=306,  # 100k steps on LJSpeech data
    use_phonemes=True,
    print_step=25,
    print_eval=True,
    mixed_precision=False,  # Mixed precision doesn't seem to be faster even with apex
    output_path=output_path,
    dataset_configs=[dataset_config],
    enable_eos_bos=True,  #
    # character_config={},
    stopnet_pos_weight=0.2,
    max_decoder_steps=1000,
)

# init model
model = Tacotron2(config)
continue_path = ''
# continue_path = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/coqui_tts-20220127_2052-febb93cf'
save_on_interrupt = True
# save_on_interrupt = False

# # init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(continue_path=continue_path, save_on_interrupt=save_on_interrupt),
    config,
    output_path,
    model=model,
    train_items=train_items,
    eval_items=eval_items,
    training_assets={"audio_processor": ap},
)
trainer.fit()
# trainer.test_run()

