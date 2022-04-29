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


def prune_model(config_path, model_file, epochs, prune_amount,
                exclude_params=('stopnet',), multiplier=0.5, new_scheduler=False, new_optimizer=False):
    C = load_config(config_path)
    ap = AudioProcessor(**C.audio)
    C.run_eval = False
    C.epochs = epochs  # PARP = prune, run on 1 epoch, prune again
    C.print_eval = False
    C.checkpoint = False

    if new_scheduler:
        C.lr_scheduler = "FixedLR"
        C.lr_scheduler_params = {"multiplier": multiplier}

    suffix = ''
    if new_optimizer:
        suffix += 'o'
    if new_scheduler:
        suffix += f's{multiplier}'
    if suffix:
        suffix = '_' + suffix
    output_path = os.path.join(BASE_DIR, 'parp',
                               f'sparsity_{int(prune_amount * 100)}{suffix}_{"_".join(exclude_params)}')
    os.makedirs(output_path, exist_ok=True)

    # load training samples
    train_items, _ = load_train_eval_items(C.dataset_configs[0], dataset_items=None)
    t = Trainer(
        TrainingArgs(save_on_finish=False, save_on_interrupt=True,
                     restore_path=model_file, exit_on_finish=False,
                     new_scheduler=new_scheduler,
                     new_optimizer=new_optimizer,
                     ),
        config=C,
        model=Tacotron2(C),
        output_path=output_path,
        train_items=train_items,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )

    global_prune(t.model, prune_amount, exclude_params)
    t.fit()
    global_prune(t.model, prune_amount, exclude_params)

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



# Keep the stopnet
# Experimented with using new scheduler, optimizer, or continuing
# Conclusion: use new scheduler but continue with loaded optimizer to stabilize initial gradients

if __name__ == '__main__':
    # prune_amount = 0.9
    # C, model = test_prune(CONFIG_PATH, MODEL_FILE, prune_amount=prune_amount)

    # save_path = os.path.join(BASE_DIR, f'sparsity_{90}.pth.tar')
    # save_model(C, model, None, None, None, 0, 0, save_path)

    for prune_amount in [0.1]: # , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        prune_model(CONFIG_PATH, MODEL_FILE, epochs=1, prune_amount=prune_amount,
                exclude_params=('stop',),
                multiplier=0.5,
                new_scheduler=True,
                new_optimizer=False,
                )

    # config, ap = get_config(ZERO_DIR)
    # loader = setup_loader(config, ap, TEST_CSV)
    # for i in range(10, 20, 10):
    #     sparsity_dir = os.path.join(BASE_DIR, f'sparsity_{i}_s0.5_stop')
    #     compute_test_mels(sparsity_dir, config, 'epoch_1.pth.tar', loader)
    # for d in ['sparsity_60_s0.5_stop', 'sparsity_60_s0.5_embed_stop']:
    #     sparsity_dir = os.path.join(BASE_DIR, d)
    #     compute_test_wavs(sparsity_dir, vocoder_file=VOCODER_ZERO_FILE, save_to_vocoder=False)


