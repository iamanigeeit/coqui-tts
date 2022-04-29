# EPIC TTS Models

This is the code for the paper **Empirical Pruning Investigations Characterizing Speech-To-Text Models (EPIC TTS Models)**.

It was forked from the 2021-10-26 version of [Coqui-TTS](https://github.com/coqui-ai/TTS) and edited. Please refer to the readme there for more details. I made some changes for more flexible pre-processing. Otherwise, the main updates in my paper are [here](/TTS/recipes/ljspeech/prune). If you wish to run it, you have to edit the relevant `*_tacotron_dca.py` files (there are no command-line args for that).

I implemented SNIP myself as there was no correct PyTorch version; it is not elegant as it requires reloading the model after computing each minibatch of gradients, but it gets around the issue of going `backward()` twice on the same graph. 

The `sparselearning` library was tweaked to make it work for RNNs.

On hindsight, i should have used FastSpeech2 as it solves many of the robustness problems i encountered with Tacotron2. The repeat / skipping in Tacotron2 (even in the baseline) creates disproportionate effects in MOS. Modifying the stopnet loss helped somewhat but could not eliminate the problems.