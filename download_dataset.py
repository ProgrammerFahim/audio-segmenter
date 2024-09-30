from torchaudio.datasets import LIBRISPEECH
downloaded_librispeech_ds = LIBRISPEECH(root="./", url="train-clean-100", download=True)