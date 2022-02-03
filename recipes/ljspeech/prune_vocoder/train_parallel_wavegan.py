import os

from TTS.vocoder.configs.parallel_wavegan_config import ParallelWaveganConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.datasets.preprocess import load_data_from_splits
from TTS.vocoder.models import setup_model

data_dir = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1"
output_dir = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune_vocoder'

config = ParallelWaveganConfig(
    train_df_file=os.path.join(data_dir, 'splits', 'train.csv'),
    val_df_file=os.path.join(data_dir, 'splits', 'val.csv'),
    feature_path=os.path.join(data_dir, 'mel'),
    output_path=output_dir,
    epochs=611,  # 400k steps
    print_step=200,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    batch_size=16,
    steps_to_start_discriminator=50000,
    print_eval=True,
    eval_split_size=32,
    lr_scheduler_gen="ExponentialLR",  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params={"gamma": 0.5**(1/100000), "last_epoch": -1},
    lr_scheduler_disc="ExponentialLR",  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params={"gamma": 0.5**(1/100000), "last_epoch": -1},
    scheduler_after_epoch=False,
    # Settings below are from the original paper
    # batch_size=8,
    stft_loss_weight=1.0,
    mse_G_loss_weight=4.0,
    # steps_to_start_discriminator=100000,
    lr_gen=0.0001,
    lr_disc=0.00005,
    # lr_scheduler_gen="StepLR",
    # lr_scheduler_gen_params={"gamma": 0.5, "step_size": 200000, "last_epoch": -1},
    # lr_scheduler_disc="StepLR",
    # lr_scheduler_disc_params={"gamma": 0.5, "step_size": 200000, "last_epoch": -1},
    # scheduler_after_epoch=False,
)

train_items, eval_items = load_data_from_splits([config.train_df_file, config.val_df_file], config.feature_path)

# setup audio processor
ap = AudioProcessor(**config.audio)

# from TTS.vocoder.datasets import GANDataset
# from torch.utils.data import DataLoader
# dataset = GANDataset(
#     ap=ap,
#     items=train_items,
#     seq_len=config.seq_len,
#     hop_len=ap.hop_length,
#     pad_short=config.pad_short,
#     conv_pad=config.conv_pad,
#     return_pairs=config.diff_samples_for_G_and_D if "diff_samples_for_G_and_D" in config else False,
#     use_noise_augment=config.use_noise_augment,
#     use_cache=config.use_cache,
# )
# dataset.shuffle_mapping()
# sampler = None
# loader = DataLoader(
#     dataset,
#     batch_size=config.batch_size,
#     shuffle=False,
#     drop_last=False,
#     sampler=sampler,
#     num_workers=config.num_loader_workers,
#     pin_memory=False,
# )
#
#
# for sample in loader:
#     break

# init the model from config
model = setup_model(config)

# continue_path = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune_vocoder/coqui_tts-20220128_0224-febb93cf/'
# restore_path = continue_path + 'checkpoint_41265.pth.tar'
# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(
        # continue_path=continue_path,
        # restore_path=restore_path,
        save_on_interrupt=True
    ),
    config,
    config.output_path,
    model=model,
    train_items=train_items,
    eval_items=eval_items,
    training_assets={"audio_processor": ap},
    parse_command_line_args=False,
)
trainer.fit()
