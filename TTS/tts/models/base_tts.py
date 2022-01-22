import os
import random
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.model import BaseModel
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_manager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import make_symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram

# pylint: skip-file


class BaseTTS(BaseModel):
    """Base `tts` class. Every new `tts` model must inherit this.

    It defines common `tts` specific functions on top of `Model` implementation.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    def _set_model_args(self, config: Coqpit):
        """Setup model args based on the config type.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign the directly.
        """
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            if "character_config" in config:
                _, self.config, num_chars = self.get_characters(config)
                self.config.num_chars = num_chars
                if hasattr(self.config, "model_args"):
                    config.model_args.num_chars = num_chars
                    self.args = self.config.model_args
            else:
                self.config = config
                self.args = config.model_args
        elif "Args" in config.__class__.__name__:
            self.args = config
        else:
            raise ValueError("config must be either a *Config or *Args")

    @staticmethod
    def get_characters(config: Coqpit):
        # TODO: implement CharacterProcessor
        if config.character_config:
            symbols, phonemes = make_symbols(**config.character_config)
        else:
            from TTS.tts.utils.text.symbols import parse_symbols, phonemes, symbols

            config.character_config = CharactersConfig(**parse_symbols())
        model_characters = phonemes if config.use_phonemes else symbols
        num_chars = len(model_characters) + getattr(config, "add_blank", False)
        return model_characters, config, num_chars

    # put in __init__ ?
    def get_speaker_manager(config: Coqpit, restore_path: str, data: List, out_path: str = None) -> SpeakerManager:
        return get_speaker_manager(config, restore_path, data, out_path)

    def init_multispeaker(self, config: Coqpit):
        """Init speaker embedding layer if `use_speaker_embedding` is True and set the expected speaker embedding
        vector dimension in the network. If model uses d-vectors, then it only sets the expected dimension.

        Args:
            config (Coqpit): Model configuration.
        """
        # set number of speakers
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        elif hasattr(config, "num_speakers"):
            self.num_speakers = config.num_speakers

        # set ultimate speaker embedding size
        if config.use_speaker_embedding or config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            print(" > Init speaker_embedding layer.")
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

    def format_batch(self, batch: Dict) -> Dict:
        """Generic batch formatting for `TTSDataset`.

        You must override this if you use a custom dataset.

        Args:
            batch (Dict): [description]

        Returns:
            Dict: [description]
        """
        # setup input batch
        char_ids = batch["char_ids"]
        id_lengths = batch["id_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        wav_path = batch["wav_path"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]

        max_id_length = torch.max(id_lengths)
        max_mel_length = torch.max(mel_lengths)

        # compute durations from attention masks
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, : id_lengths[idx], : mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([id_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert (
                    dur.sum() == mel_lengths[idx]
                ), f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                durations[idx, : id_lengths[idx]] = dur

        # set stop targets wrt reduction factor
        stop_targets = stop_targets.view(char_ids.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "char_ids": char_ids,
            "id_lengths": id_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_id_length": int(max_id_length),
            "max_spec_length": int(max_mel_length),
            "wav_path": wav_path,
            "waveform": waveform,
            "pitch": pitch,
        }

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        data_items: List,
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            return None

        if is_eval:
            batch_group_size = 0
            use_noise_augment = False
            num_workers = config.num_eval_loader_workers
            batch_size = config.eval_batch_size
            data_split = "eval_dataset"
        else:
            batch_group_size = config.batch_group_size * config.batch_size
            use_noise_augment = config.use_noise_augment
            num_workers = config.num_loader_workers
            batch_size = config.batch_size
            data_split = "train_dataset"
        ap = assets["audio_processor"]

        # setup multi-speaker attributes
        if hasattr(self, "speaker_manager") and self.speaker_manager is not None:
            speaker_id_mapping = self.speaker_manager.speaker_ids if config.use_speaker_embedding else None
            d_vector_mapping = self.speaker_manager.d_vectors if config.use_d_vector_file else None
        else:
            speaker_id_mapping = None
            d_vector_mapping = None

        # init dataset
        if hasattr(self, data_split):
            dataset = getattr(self, data_split)
        else:
            dataset = TTSDataset(
                dataset_config=config.dataset_configs[0],  # TODO: figure out how to combine datasets later
                ap=ap,
                items=data_items,
                outputs_per_step=config.r if "r" in config else 1,
                use_symbols=config.use_symbols,
                use_phonemes=config.use_phonemes,
                use_mel=config.use_mel,
                use_linear=config.use_linear,
                use_wav=config.use_wav,
                use_f0=config.use_f0,
                batch_group_size=batch_group_size,
                min_seq_len=config.min_seq_len,
                max_seq_len=config.max_seq_len,
                enable_eos_bos=config.enable_eos_bos,
                use_noise_augment=use_noise_augment,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping if config.use_d_vector_file else None,
                verbose=verbose,
            )
            # sort input sequences from short to long
            dataset.sort_and_filter_items(config.get("sort_by_audio_len", default=False))

            setattr(self, data_split, dataset)


        # sampler for DDP
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None

        # init dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        )
        return loader

    def _get_test_aux_input(
        self,
    ) -> Dict:

        d_vector = None
        if self.config.use_d_vector_file:
            d_vector = [self.speaker_manager.d_vectors[name]["embedding"] for name in self.speaker_manager.d_vectors]
            d_vector = (random.sample(sorted(d_vector), 1),)

        aux_inputs = {
            "speaker_id": None
            if not self.config.use_speaker_embedding
            else random.sample(sorted(self.speaker_manager.speaker_ids.values()), 1),
            "d_vector": d_vector,
            "style_wav": None,  # TODO: handle GST style input
        }
        return aux_inputs

    def test_run(self, assets: Dict) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        ap = assets["audio_processor"]
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            outputs_dict = synthesis(
                self,
                sen,
                self.config,
                "cuda" in str(next(self.parameters()).device),
                ap,
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                enable_eos_bos=self.config.enable_eos_bos,
                use_griffin_lim=True,
                do_trim_silence=False,
            )
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"], ap, output_fig=False
            )
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False
            )
        return test_figures, test_audios

    def on_init_start(self, trainer):
        """Save the speaker.json at the beginning of the training. And update the config.json with the
        speakers.json file path."""
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.json")
            self.speaker_manager.save_speaker_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            # some models don't have `model_args` set
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            print(f" > `speakers.json` is saved to {output_path}.")
            print(" > `speakers_file` is updated in the config.json.")
