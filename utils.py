import torch
from pathlib import Path

def load_utils(optims: list, scheds: list, save_dir: str, epoch_end: int):
    if epoch_end is None:
        raise ValueError("EPOCH_END: Specify a valid epoch to load from")

    audio_checkpoint = torch.load(Path(save_dir) / f'audio_utils_epoch_{epoch_end}.pth')
    text_checkpoint = torch.load(Path(save_dir) / f'text_utils_epoch_{epoch_end}.pth')

    optims[0].load_state_dict(audio_checkpoint["optimizer"])
    scheds[0].load_state_dict(audio_checkpoint["scheduler"])

    optims[1].load_state_dict(text_checkpoint["optimizer"])
    scheds[1].load_state_dict(text_checkpoint["scheduler"])

    return *optims, *scheds

def save_utils(optims: list, scheds: list, save_dir: str, epoch_end: int):
    torch.save({
        "optimizer": optims[0].state_dict(),
        "scheduler": scheds[0].state_dict(),
    }, Path(save_dir) / f'audio_utils_epoch_{epoch_end}.pth')

    torch.save({
        "optimizer": optims[1].state_dict(),
        "scheduler": scheds[1].state_dict(),
    }, Path(save_dir) / f'text_utils_epoch_{epoch_end}.pth')