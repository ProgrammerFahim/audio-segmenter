import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path

class TextEncoder(nn.Module):
    """
    Text Encoder for projecting text to the same embedding space as audio.
    Uses a pretrained transformer encoder (probably either Bert or Distilbert).

    Initialization:
        - encoder_name: Name of pretrained encoder to download from HuggingFace
        - encoder_path: Path to finetuned encoder to load from
            - encoder_path takes precedence over encoder_name. No need to use
              huggingface pretrained weights if we have a finetuned model
        - pooler: Whether the model has a pooler_output (distilbert doesn't)
        - encoder_hidden_dim: Size of the output of encoder
        - final_proj_dim: Size of the final audio-text projection

    Input:
        - **input: input_ids, attention_mask [, token_type_ids for bert]

    Output:
        - out: [batches, final_proj_dim]
    """
    def __init__(
        self,
        encoder_name: str = None,
        encoder_path: str = None,
        last_epoch: int = None,
        pooler: bool = True,
        encoder_hidden_dim: int = 768,
        final_proj_dim: int = 512,
    ):
        super().__init__()
        self.pooler = pooler

        if encoder_path is not None:
            if last_epoch is None:
                raise ValueError("LAST_EPOCH: Need an epoch number to load model")
            
            print("Initializing text encoder from local finetuned model...")
            self.encoder = AutoModel.from_pretrained(Path(encoder_path) / f'hf_encoder_epoch_{last_epoch}')
            self.proj = nn.Linear(encoder_hidden_dim, final_proj_dim)
            self.proj.load_state_dict(torch.load(Path(encoder_path) / f'text_proj_epoch_{last_epoch}.pth'))
        else:
            if encoder_name is None:
                raise ValueError("Either encoder_name or encoder_path must be provided")

            print("Initializing text encoder from pretrained huggingface model...")
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.proj = nn.Linear(encoder_hidden_dim, final_proj_dim)

    def forward(self, **input):
        x = self.encoder(**input)
        if self.pooler:
            x = x.pooler_output
        else:
            x = x.last_hidden_state[:, 0]
        return self.proj(x)

    def save_model(self, save_path: str, epoch_end: int):
        self.encoder.save_pretrained(Path(save_path) / f'hf_encoder_epoch_{epoch_end}')
        torch.save(self.proj.state_dict(), Path(save_path) / f'text_proj_epoch_{epoch_end}.pth')