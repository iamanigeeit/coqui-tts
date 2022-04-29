import inspect
import os
import time
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import save_model
from TTS.config import load_config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.trainer import Trainer, TrainingArgs
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from TTS.tts.datasets import load_train_eval_items
from TTS.utils.trainer_utils import will_prune
import json
from TTS.utils.generic_utils import get_module_by_name
import torch
from recipes.ljspeech.prune.train_tacotron_dca import config, train_items, eval_items, ap
from recipes.ljspeech.prune.gen_results import setup_loader

DATA_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/LJSpeech-1.1/"
SNIP_DIR = "/home/perry/PycharmProjects/TTS/recipes/ljspeech/prune/tacotron2_nomask/snip/"

os.makedirs(SNIP_DIR, exist_ok=True)
STATE_DICT_PATH = os.path.join(SNIP_DIR, 'state_dict.pt')
TOTAL_GRADS_PATH = os.path.join(SNIP_DIR, 'total_grads.pt')
MASKS_PATH = os.path.join(SNIP_DIR, 'masks.pt')


def load_model(config, state_dict):
    model = Tacotron2(config).cuda()
    model.load_state_dict(state_dict)
    return model


def mask_params(model, params_to_prune):
    fmn = [(full_name, *full_name.rsplit('.', 1)) for full_name in params_to_prune]
    fmn = [(full_name, get_module_by_name(model, m), n) for full_name, m, n in fmn]
    masks = []
    for full_name, module, name in fmn:
        param = getattr(module, name)
        mask = torch.ones_like(param, requires_grad=True).cuda()
        param_masked = param * mask
        with torch.no_grad():
            delattr(module, name)
            setattr(module, name, param_masked)
        masks.append(mask)
        assert mask.is_leaf and not getattr(module, name).is_leaf
    return masks


def create_masks(sparsity, total_grads, masks_path=MASKS_PATH):
    flattened_grads = torch.cat([total_grad.view(-1) for total_grad in total_grads.values()])
    threshold = torch.kthvalue(flattened_grads, int(sparsity * len(flattened_grads))).values.item()
    masks = {}
    for full_name, total_grad in total_grads.items():
        mask = total_grad > threshold
        masks[full_name] = mask
        print(f'{full_name}: {mask.numel()} -> {mask.sum().item()}')
    fullpath, ext = os.path.splitext(masks_path)
    save_path = f'{fullpath}_sparsity_{int(100 * sparsity)}{ext}'
    torch.save((sparsity, masks), save_path)
    print(f'Masks saved to {save_path}')
    return masks


def masks_from_grads_file(sparsity, total_grads_path=TOTAL_GRADS_PATH, masks_path=MASKS_PATH):
    total_grads = torch.load(total_grads_path)
    masks = create_masks(sparsity, total_grads, masks_path)
    return masks


def compute_gradients(config, data_loader=None,
                      state_dict_path=STATE_DICT_PATH,
                      total_grads_path=TOTAL_GRADS_PATH,
                      exclude_params=('stop', 'batch_norm')):

    if os.path.exists(total_grads_path):
        print(f'Loading gradients from {total_grads_path}')
        return torch.load(total_grads_path)

    if os.path.exists(state_dict_path):
        print(f'Loading state_dict from {state_dict_path}')
        state_dict = torch.load(state_dict_path)
        model = load_model(config, state_dict)
    else:
        model = Tacotron2(config).cuda()
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_path)
        print(f'Saved state_dict to {state_dict_path}')

    start = time.time()
    assert data_loader
    param_sizes = {full_name: p.numel() for full_name, p in model.named_parameters() if p.requires_grad}
    print('Total trainable params: {}'.format(sum(param_sizes.values())))
    params_to_prune = [full_name for full_name, _ in model.named_parameters()
                       if will_prune(param_name=full_name, exclude_params=exclude_params)]
    print('Total params eligible to prune: {}'.format(sum(param_sizes[full_name] for full_name in params_to_prune)))

    masks = mask_params(model, params_to_prune)
    total_grads = [torch.zeros_like(mask).cuda() for mask in masks]

    for i, batch in enumerate(data_loader):
        print(f'Computing gradients for batch {i}...')
        criterion = model.get_criterion().cuda()
        batch = model.format_batch(batch)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()
        model.zero_grad()
        _, loss_dict = model.train_step(batch, criterion)
        loss = loss_dict['loss']
        grads = torch.autograd.grad(loss, masks)
        total_grads = [total_grad + grad for total_grad, grad in zip(total_grads, grads)]
        model = load_model(model.config, state_dict)
        masks = mask_params(model, params_to_prune)

    total_grads = {full_name: total_grad.abs() for full_name, total_grad in zip(params_to_prune, total_grads)}
    torch.save(total_grads, total_grads_path)
    print(f'Saved total_grads to {total_grads_path}')

    time_taken = time.time() - start
    print(f'SNIP time: {time_taken}')
    print(f'SNIP time/batch: {time_taken / (i+1)}')
    return total_grads




if __name__ == '__main__':

    # data_loader = setup_loader(config, ap, os.path.join(DATA_DIR, 'splits', 'train.csv'))
    total_grads = compute_gradients(config, data_loader=None)
    # create_masks(sparsity=0.2, total_grads=total_grads)
    # create_masks(sparsity=0.4, total_grads=total_grads)
    create_masks(sparsity=0.6, total_grads=total_grads)

    fullpath, ext = os.path.splitext(MASKS_PATH)
    masks_path = f'{fullpath}_sparsity_60{ext}'
    # init model
    state_dict = torch.load(STATE_DICT_PATH)
    model = load_model(config, state_dict)
    save_on_interrupt = True

    config.epochs = 100
    # # init the trainer and 🚀
    trainer = Trainer(
        TrainingArgs(save_on_interrupt=save_on_interrupt, snip_mask_path=masks_path),
        config,
        model=model,
        train_items=train_items,
        eval_items=eval_items,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )
    trainer.fit()
    # trainer.test_run()