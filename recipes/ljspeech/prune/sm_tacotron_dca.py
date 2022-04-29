import inspect
import os
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import save_model
from TTS.config import load_config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.datasets import load_train_eval_items
from TTS.utils.trainer_utils import global_prune
import json
from gen_results import ZERO_DIR, TEST_CSV, VOCODER_ZERO_FILE, compute_test_mels, compute_test_wavs, get_config, setup_loader


BASE_DIR = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask/'
MODEL_DIR = BASE_DIR + "baseline/"
CONFIG_PATH = MODEL_DIR + "config.json"
STEPS = 100000
MODEL_FILE = MODEL_DIR + f"checkpoint_{STEPS}.pth.tar"
DATA_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/"


def prune_model(config_path, model_file, epochs, prune_amount, exclude_params=('stopnet',),
                lr_multiplier=0.5, sm_prune_rate=0.2, growth_mode='momentum', prune_every_k=0,
                new_scheduler=False, new_optimizer=False):
    C = load_config(config_path)
    ap = AudioProcessor(**C.audio)
    C.run_eval = False
    C.epochs = epochs
    C.print_eval = False
    C.checkpoint = False

    if new_scheduler:
        C.lr_scheduler = "FixedLR"
        C.lr_scheduler_params = {"multiplier": lr_multiplier}

    suffix = ''
    if new_optimizer:
        suffix += 'o'
    if new_scheduler:
        suffix += f's{lr_multiplier}'
    if suffix:
        suffix = '_' + suffix
    output_path = os.path.join(BASE_DIR, 'sm',
                               f'sparsity_{int(prune_amount * 100)}{suffix}_{"_".join(exclude_params)}'
                               f'_p{sm_prune_rate}_k_{prune_every_k}_{growth_mode}'
                               )

    # load training samples
    train_items, _ = load_train_eval_items(C.dataset_configs[0], dataset_items=None)
    t = Trainer(
        TrainingArgs(save_on_finish=False, save_on_interrupt=True,
                     restore_path=model_file, exit_on_finish=False,
                     new_scheduler=new_scheduler,
                     new_optimizer=new_optimizer,
                     sm_sparsity=prune_amount,
                     exclude_params=exclude_params,
                     sm_prune_rate=sm_prune_rate,
                     sm_mode='resume',
                     sm_growth_mode=growth_mode,
                     sm_prune_every_k=prune_every_k,
                     ),
        config=C,
        model=Tacotron2(C),
        output_path=output_path,
        train_items=train_items,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )

    t.fit()

    save_path = os.path.join(t.output_path, f'epoch_{epochs}.pth.tar')
    save_model(t.config, t.model, t.optimizer, t.scheduler, t.scaler, t.total_steps_done, t.epochs_done, save_path)
    print(f'Saved pruned model to {save_path}')

    with open(os.path.join(output_path, 'prune_config.json'), 'w') as f:
        json.dump({
            'new_scheduler': new_scheduler,
            'new_optimizer': new_optimizer,
            'scheduler': C.lr_scheduler,
            'scheduler_params': C.lr_scheduler_params,
            'exclude_params': list(exclude_params),
        }, f)


def sparse_training(prune_amount, exclude_params=('stopnet',), sm_prune_rate=0.2, sm_mode='constant',
                    continue_path='', save_on_interrupt=True):
    from train_tacotron_dca import config, train_items, eval_items, ap
    model = Tacotron2(config)
    trainer = Trainer(
        TrainingArgs(continue_path=continue_path,
                     save_on_interrupt=save_on_interrupt,
                     sm_sparsity=prune_amount,
                     exclude_params=exclude_params,
                     sm_prune_rate=sm_prune_rate,
                     sm_mode=sm_mode,
                     ),
        config,
        model=model,
        train_items=train_items,
        eval_items=eval_items,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )
    trainer.fit()

# When using pretrained model, use resume for the sparse momentum mode, because to reinit the weights makes no sense.

if __name__ == '__main__':
    for prune_amount in [0.2]: #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        sparse_training(prune_amount)

    # for prune_amount in [0.4]: #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #     prune_model(CONFIG_PATH, MODEL_FILE, epochs=1, prune_amount=prune_amount,
    #                 exclude_params=('stop',),
    #             new_scheduler=True,
    #             new_optimizer=False,
    #             prune_every_k=50,
    #             )


