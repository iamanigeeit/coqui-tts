import torch
import numpy as np
import os
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import save_model
from TTS.config import load_config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.trainer import Trainer, TrainingArgs
from TTS.vocoder.datasets.preprocess import load_data_from_splits
from TTS.utils.trainer_utils import global_prune
from TTS.vocoder.models import setup_model

BASE_DIR = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune_vocoder/coqui_tts-20220214_2137-aa986857/'
DATA_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/"
TEST_CSV = DATA_DIR + "splits/test.csv"
MODEL_DIR = BASE_DIR + "sparsity_0/"
CONFIG_PATH = MODEL_DIR + "config.json"
STEPS = 200000
MODEL_FILE = MODEL_DIR + f"checkpoint_{STEPS}.pth.tar"



def prune_model(config_path, model_file, epochs, prune_amount):

    c = load_config(config_path)
    ap = AudioProcessor(**c.audio)
    c.run_eval = False
    c.epochs = epochs  # PARP = prune, run on 1 epoch, prune again
    c.print_eval = False
    c.lr_scheduler_gen = "FixedLR"
    c.lr_scheduler_disc = "FixedLR"
    c.lr_scheduler_gen_params = {"multiplier": 0.5}
    c.lr_scheduler_disc_params = {"multiplier": 0.5}
    c.checkpoint = False
    c.feature_path = os.path.join(DATA_DIR, 'mel_norm')

    # load training samples
    train_items = load_data_from_splits(c.train_df_file, c.feature_path)

    t = Trainer(
        TrainingArgs(save_on_finish=False, save_on_interrupt=True,
                     restore_path=model_file, exit_on_finish=False,
                     new_scheduler=True),
        config=c,
        model=setup_model(c),
        output_path=os.path.join(os.path.dirname(model_file), f'sparsity_{int(prune_amount * 100)}'),
        train_items=train_items,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )
    global_prune(t.model.model_g, prune_amount)

    for submodule in t.model.modules():
        submodule.register_forward_hook(nan_hook)

    # torch.autograd.set_detect_anomaly(True)
    t.fit()
    global_prune(t.model.model_g, prune_amount)

    save_path = os.path.join(t.output_path, f'epoch_{epochs}.pth.tar')
    save_model(t.config, t.model, t.optimizer, t.scheduler, t.scaler, t.total_steps_done, t.epochs_done, save_path)


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            nan_positions = nan_mask.nonzero()
            # problem_tensor = out.detach().cpu().numpy()
            # np.save('/tmp/problem.npy', problem_tensor)
            # for j, input_tensor in enumerate(inp):
            #     print(torch.isnan(input_tensor).any())
            #     problem_inp = input_tensor.detach().cpu().numpy()
            #     np.save(f'/tmp/problem_inp{j}.npy', problem_inp)
            # torch.save(self.state_dict(), '/tmp/conv1d.pt')
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_positions, "where:",
                               out[nan_positions[:, 0].unique(sorted=True)])


if __name__ == '__main__':
    for prune_amount in [0.1*x for x in range(1, 2)]:
        prune_model(CONFIG_PATH, MODEL_FILE, epochs=1, prune_amount=prune_amount)
