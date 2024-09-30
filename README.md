# Audio Segmenter

Inspired by [GroupVit](https://arxiv.org/abs/2202.11094), this repository contains code
that tries to adapt the architecture for audio segmentation. Concisely, given an audio
sample and a word uttered within it, this model is supposed to encode in the grouping
blocks's attention matrices the mask that points to where that word is present within
the audio sample.

To train the model, first enter your WANDB API key in the `config/config.py` file:

```
WANDB_API_KEY = "<your_api_key>"
```

Then, download the LibriSpeech training dataset:

```
python download_dataset.py
```

Then, just run `train.py`:

```
python train.py
```
