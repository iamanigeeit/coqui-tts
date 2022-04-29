import inspect
import os
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import save_model
from TTS.config import load_config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.datasets import load_train_eval_items
from TTS.tts.models import setup_model as setup_tts_model
from TTS.utils.trainer_utils import global_prune
import torch
from TTS.utils.generic_utils import get_module_by_name, get_param_by_name, count_parameters
from torch.nn.utils import prune

BASE_DIR = '/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask/'
MODEL_DIR = BASE_DIR + "baseline/"
CONFIG_PATH = MODEL_DIR + "config.json"
STEPS = 100000
MODEL_FILE = MODEL_DIR + f"checkpoint_{STEPS}.pth.tar"
DATA_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/"
UMP_DIR = BASE_DIR + 'ump/'
TRIM_DIR = BASE_DIR + 'trim/'

PARAM_PRUNE_DIM = {'embedding.weight': 1,
                   'encoder.convolutions.0.convolution1d.weight': 0,
                   'encoder.convolutions.1.convolution1d.weight': 0,
                   'encoder.convolutions.2.convolution1d.weight': 0,
                   'encoder.lstm.weight_ih_l0': 0,
                   'encoder.lstm.weight_hh_l0': 0,
                   'encoder.lstm.weight_ih_l0_reverse': 0,
                   'encoder.lstm.weight_hh_l0_reverse': 0,
                   'decoder.prenet.linear_layers.0.linear_layer.weight': 0,
                   'decoder.attention_rnn.weight_ih': 0,
                   'decoder.attention_rnn.weight_hh': 0,
                   'decoder.attention.query_layer.weight': 0,
                   'decoder.attention.key_layer.weight': 0,
                   'decoder.decoder_rnn.weight_ih': 0,
                   'decoder.decoder_rnn.weight_hh': 0,
                   'postnet.convolutions.0.convolution1d.weight': 0,
                   'postnet.convolutions.1.convolution1d.weight': 0,
                   'postnet.convolutions.2.convolution1d.weight': 0,
                   'postnet.convolutions.3.convolution1d.weight': 0,
                   'coarse_decoder.prenet.linear_layers.0.linear_layer.weight': 0,
                   'coarse_decoder.prenet.linear_layers.1.linear_layer.weight': 0,
                   'coarse_decoder.attention_rnn.weight_ih': 0,
                   'coarse_decoder.attention_rnn.weight_hh': 0,
                   'coarse_decoder.attention.query_layer.weight': 0,
                   'coarse_decoder.attention.key_layer.weight': 0,
                   'coarse_decoder.decoder_rnn.weight_ih': 0,
                   'coarse_decoder.decoder_rnn.weight_hh': 0,
                    }

def prune_multiple(sparsities, output_dir,
                   config_path=CONFIG_PATH, model_file=MODEL_FILE, steps=STEPS, exclude_params=('stop',)):
    config = load_config(config_path)
    for sparsity in sparsities:
        sparsity_dir = os.path.join(output_dir, f'sparsity_{int(sparsity * 100)}_{"_".join(exclude_params)}')
        os.makedirs(sparsity_dir, exist_ok=True)
        output_path = os.path.join(sparsity_dir, os.path.basename(model_file))
        model = setup_tts_model(config)
        model.load_checkpoint(config, model_file, eval=True)
        model = model.cuda()
        global_prune(model, sparsity, exclude_params)
        save_model(config, model,
                   optimizer=None,
                   scheduler=None,
                   scaler=None,
                   current_step=steps,
                   epoch=None,
                   output_path=output_path)
        for name, param in model.named_parameters():
            print(name, param.numel(), '->', param.count_nonzero().item())
        print(f'Saved to {output_path}')


def trim_multiple(sparsities, ump_sparsities, output_dir, ump_dir=UMP_DIR,
                  config_path=CONFIG_PATH, model_file=MODEL_FILE, steps=STEPS, param_prune_dim=PARAM_PRUNE_DIM):

    config = load_config(config_path)

    for sparsity, ump_sparsity in zip(sparsities, ump_sparsities):
        model = setup_tts_model(config)
        model.load_checkpoint(config, model_file, eval=True)
        model = model.cuda()

        ump_file = os.path.join(ump_dir, f'sparsity_{int(ump_sparsity * 100)}_stop/checkpoint_{steps}.pth.tar')
        ump_model = setup_tts_model(config)
        ump_model.load_checkpoint(config, ump_file, eval=True)
        ump_model = ump_model.cuda()
        ump_params = {n: (p.numel(), p.count_nonzero().item()) for n, p in ump_model.named_parameters()}

        units_to_prune = {}
        for name in param_prune_dim:
            numel, nonzero = ump_params[name]
            param_sparsity = (1 - nonzero / numel) / ump_sparsity * sparsity
            units_to_prune[name] = param_sparsity

        for name, param_sparsity in units_to_prune.items():
            m, n = name.rsplit('.', 1)
            module = get_module_by_name(model, m)
            dim = param_prune_dim[name]
            param = getattr(module, n)
            print(f'Pruning {name}: {param.shape[dim]} channels at ratio {param_sparsity}')
            if param.shape[dim] * param_sparsity > 0.5:
                prune.ln_structured(module, n, param_sparsity, 1, dim)
                prune.remove(module, n)
                bias_name = n.replace('weight', 'bias')
                if hasattr(module, bias_name):
                    bias = getattr(module, bias_name)
                    if bias is not None:
                        print(f'Pruning {m}.{bias_name}')
                        add_axes = list(range(param.dim()))
                        add_axes.pop(dim)
                        param_sum = param.abs().sum(axis=add_axes)
                        bias = bias * (param_sum > 0.0)
                        bias = torch.nn.Parameter(bias)
                        setattr(module, bias_name, bias)

        total_removed = 0
        for n, p in model.named_parameters():
            numel = p.numel()
            nonzeros = p.count_nonzero().item()
            total_removed += (numel - nonzeros)
            print(n, numel, '->', nonzeros)

        print('Net sparsity', total_removed / count_parameters(model))

        output_path = os.path.join(output_dir, f'sparsity_{sparsity*100}', f'checkpoint_{steps}.pth.tar')
        save_model(config, model,
                           optimizer=None,
                           scheduler=None,
                           scaler=None,
                           current_step=steps,
                           epoch=None,
                           output_path=output_path)
        print(f'Saved to {output_path}')


if __name__ == '__main__':
    # prune_multiple(sparsities=[0.1*x for x in range(1, 7)],
    #                output_dir=UMP_DIR)

    trim_multiple(sparsities=[0.041], ump_sparsities=[0.4],
                  output_dir=TRIM_DIR)

    # trim_multiple(sparsities=[0.051], ump_sparsities=[0.5],
    #               output_dir=TRIM_DIR)

