import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from functools import partial
from pathlib import Path

from dataset.dataset import ASRDataset, collate_fn_with_tokenizer
from models.audio_model import AudioSegmenter
from models.text_model import TextEncoder
from utils import load_utils, save_utils
from config.audio_config import SPEC_CONFIG
from config.text_config import ENCODER_NAME, POOLER
from config.config import *
from loss import compute_loss

Path(UTILS_SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(AUDIO_ENCODER_SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(TEXT_ENCODER_SAVE_PATH).mkdir(parents=True, exist_ok=True)

wandb.login(key=WANDB_API_KEY)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

text_tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME, clean_up_tokenization_spaces=True)
dataset = ASRDataset('train_100_dataset.csv', SPEC_CONFIG)
collate_fn = partial(collate_fn_with_tokenizer, tokenizer=text_tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)

if FINETUNING:
    text_encoder_config = {"encoder_path": TEXT_ENCODER_SAVE_PATH, "pooler": POOLER, "last_epoch": LAST_EPOCH}
    segmenter_config = {"encoder_path": AUDIO_ENCODER_SAVE_PATH, "last_epoch": LAST_EPOCH}
else:
    text_encoder_config = {"encoder_name": ENCODER_NAME, "pooler": POOLER}
    segmenter_config = {"encoder_path": None}

audio_encoder = AudioSegmenter(**segmenter_config).to(DEVICE)
text_encoder = TextEncoder(**text_encoder_config).to(DEVICE)

# Set up audio optimizer and scheduler
# Load optimizer state_dict after initialization when necessary
audio_optimizer = Adam(audio_encoder.parameters(), lr=0.0001, weight_decay=0.05)
audio_scheduler = CosineAnnealingLR(audio_optimizer, T_max=COSINE_SCHED_T_MAX)

# Set up text optimizer and scheduler
text_optimizer = AdamW(text_encoder.parameters(), lr=2e-5, eps=1e-8)
text_scheduler = get_linear_schedule_with_warmup(text_optimizer, 0.1, len(dataloader) * EPOCHS)

run = wandb.init(project=PROJECT_NAME,)

audio_encoder.train()
text_encoder.train()

for p in text_encoder.parameters():
    p.requires_grad = False

if FINETUNING:
    audio_optimizer, text_optimizer, audio_scheduler, text_scheduler = load_utils(
        optims=[audio_optimizer, text_optimizer],
        scheds=[audio_scheduler, text_scheduler],
        save_dir=UTILS_SAVE_PATH, epoch_end=LAST_EPOCH)

total_steps = len(dataloader)
total_batches = np.ceil(total_steps / GRAD_ACCUMULATION_STEPS)
print(f"Total steps: {total_steps}, total batches: {total_batches}")

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch+1}/{EPOCHS}, Total steps in epoch: {total_steps}")

    step = 0
    batch = 0
    batch_loss = 0.
    for x, src_padding_mask, transcript, word_mat in tqdm(dataloader):
        x = x.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        transcript = {k: v.to(DEVICE) for k, v in transcript.items()}
        word_mat = word_mat.to(DEVICE)

        audio_proj = audio_encoder(x, src_padding_mask)[0].squeeze(1)               # [batch, proj]
        text_proj = text_encoder(**transcript)                                      # [batch + batch_vocab, proj]

        audio_proj = F.normalize(audio_proj, p=2.0, dim=1)
        text_proj = F.normalize(text_proj, p=2.0, dim=1)

        loss, loss_components = compute_loss(audio_proj, text_proj, word_mat)
        loss = loss / GRAD_ACCUMULATION_STEPS
        batch_loss += loss.item()
        loss.backward()

        if ((step + 1) % LOG_EVERY_N_STEPS == 0):
            wandb.log({
                "audio_loss": loss_components[0].item(),
                "text_loss": loss_components[1].item(),
                "vocab_loss": loss_components[2].item(),
                "loss": batch_loss,
            })
            tqdm.write(f"epoch: {epoch+1}, batch: {batch+1}/{total_batches}, loss: {batch_loss}")

        if ((step + 1) % GRAD_ACCUMULATION_STEPS == 0) or ((step + 1) == total_steps):
            audio_optimizer.step()
            text_optimizer.step()
            audio_optimizer.zero_grad()
            text_optimizer.zero_grad()
            text_scheduler.step()
            batch += 1
            batch_loss = 0.

        step += 1

    audio_scheduler.step()

    audio_encoder.save_model(AUDIO_ENCODER_SAVE_PATH, epoch_end=epoch)
    text_encoder.save_model(TEXT_ENCODER_SAVE_PATH, epoch_end=epoch)

    save_utils(
        optims=[audio_optimizer, text_optimizer],
        scheds=[audio_scheduler, text_scheduler],
        save_dir=UTILS_SAVE_PATH, epoch_end=epoch)

wandb.finish()
