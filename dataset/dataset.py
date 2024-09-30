import torch
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset


class ASRDataset(Dataset):
    def __init__(self, source_file, spec_config):
        self.audio_list = pd.read_csv(source_file)
        in_range = self.audio_list["duration"] < 16.0
        self.audio_list = self.audio_list[in_range]
        self.audio_list.index = range(len(self.audio_list))
        self.spec_config = spec_config

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_file = self.audio_list["path"][idx]
        audio, _ = librosa.load(audio_file, sr=self.spec_config["sr"])

        melspec = librosa.feature.melspectrogram(y=audio, **self.spec_config)

        return (melspec, self.audio_list["transcript"][idx])


def collate_fn_with_tokenizer(data, tokenizer):
    batches = len(data)
    audios, transcripts = zip(*data)

    n_features = audios[0].shape[0]
    max_len = np.max([x.shape[1] for x in audios])

    batch = np.zeros((batches, n_features, max_len))
    mask = np.ones((batches, max_len))

    for idx, audio in enumerate(audios):
        batch[idx, :, :audio.shape[1]] = audio
        mask[idx, :audio.shape[1]] = 0

    words = [word for line in transcripts for word in line.split(" ")]
    words = set(words)

    word_mat = torch.tensor([
        [word in line.split(" ") for word in words]
        for line in transcripts
    ], dtype=torch.float32)

    transcripts = list(transcripts) + list(words)
    transcripts = tokenizer(transcripts, padding="longest", return_tensors="pt")

    return (
        torch.from_numpy(batch).to(torch.float32).transpose(-1, -2),
        torch.from_numpy(mask).type(torch.bool),
        transcripts,
        word_mat,
    )