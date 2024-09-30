import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat
from pathlib import Path

class GroupingLayer(nn.Module):
    """
    A Grouping Layer at the end of every Grouping Stage takes as input group tokens and
    sequence tokens, makes the group tokens (query) attend to the sequence tokens (key)
    and returns the modified group tokens. To make each group token attend to certain
    sequence tokens discretely, the attention weights are generated using Gumbel Softmax.

    Initialization:
        - num_groups: Number of group tokens present in the input to forward
        - embed_dim: Size of the embeddings of the group and sequence tokens
        - gumbel: Whether to use gumbel softmax for attention weights
        - gumbel_tau: Value of the tau parameter in Gumbel-Softmax
        - hard: Whether to use hard attention (1s and 0s) or soft attention (probabilities)
        - bias: Whether to use bias in the linear projections layers

    Inputs:
        - x: [batches, grp_tokens + seq_tokens, embed_dim]
        - src_padding_mask and return_attn: Same as Grouping Stage

    Outputs:
        - group_embs: Modified group tokens [batches, grp_tokens, embed_dim]
        - group_assn: Attention weights [batches, grp_tokens, seq_tokens]
    """
    def __init__(
        self,
        num_groups: int,
        embed_dim: int = 384,
        gumbel: bool = True,
        gumbel_tau: float = 1.,
        hard: bool = True,
        bias: bool = False,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau

        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, x, src_padding_mask=None, return_attn=False):
        # shape of x: [batches, grp_toks + seq_toks, embed_dim]
        group_toks, seq_toks = x[:, :self.num_groups], x[:, self.num_groups:]

        query = self.W_q(group_toks)
        key = self.W_k(seq_toks)
        value = self.W_v(seq_toks)

        attention = einsum(query, key, "b g d, b s d -> b g s")

        group_assn = self.get_group_assn(attention, dim=-2)

        if src_padding_mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html
            group_assn.masked_fill_(src_padding_mask.unsqueeze(1), 0.)

        group_embs = einsum(group_assn, value, "b g s, b s d -> b g d")
        out = self.norm(self.dropout(self.W_o(group_embs)) + group_toks)

        if return_attn:
            soft_assn = F.softmax(attention, dim=-2)
            soft_assn = soft_assn.masked_fill(src_padding_mask.unsqueeze(1), 0.)
            attn_dict = {"attn": group_assn, "soft": soft_assn}
        else:
            attn_dict = None

        return out, attn_dict

    def get_group_assn(self, attn, dim):
        """Compute group assignment matrix"""
        if self.gumbel and self.training:
            assn = F.gumbel_softmax(attn, tau=self.gumbel_tau, hard=self.hard, dim=dim)
        elif self.hard:
            assn = self.hard_softmax(attn, dim)
        else:
            assn = F.softmax(attn, dim=dim)

        return assn

    def hard_softmax(self, logits, dim):
        y_soft = logits.softmax(dim)

        # Straight through trick
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft


class GroupingStage(nn.Module):
    """
    A Grouping Stage consists of a number of transformer encoder layers followed by a
    Grouping Layer. The whole stage takes as input the sequence tokens, and appends the
    group tokens to it in the front. The output of the Grouping Stage are the modified
    group tokens. The encoder layers bring information from the sequence tokens to the
    group tokens, and the Grouping Layer uses that to determine which sequence tokens
    should be attended to by the group tokens.

    Initialization:
        - num_groups: Number of group tokens present in this stage
        - num_layers: Number of transformer encoder layers in this stage
        - embed_dim: Size of the embeddings of the group and sequence tokens
        - num_heads: Number of attention heads in the transformer encoder layers
        - mlp_scale: The factor by which the MLP hidden layer differs from embed_dim
        - transformer_activation: The activation function used in the encoder MLPs
        - layer_norm_eps: Epsilon value for LayerNorm layers
        - other params: Same as Grouping Layer

    Inputs:
        - x, src_padding_mask, return_attn: Same as AudioSegmenter

    Outputs:
        - group_embs and group_assn: Same as Grouping Layer
    """
    def __init__(
        self,
        num_groups: int,
        num_layers: int,
        embed_dim: int = 384,
        gumbel: bool = True,
        gumbel_tau: float = 1.,
        hard: bool = True,
        grouping_bias: bool = False,
        num_heads: int = 8,
        mlp_scale: int = 4,
        transformer_activation: str = "relu",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = num_groups

        self.groups = nn.Parameter(torch.randn(num_groups, embed_dim))

        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_scale,
                dropout=dropout,
                activation=transformer_activation,
                batch_first=True,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=num_layers,
        )

        self.grouping_layer = GroupingLayer(
            num_groups=num_groups,
            embed_dim=embed_dim,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            hard=hard,
            bias=grouping_bias,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )

    def forward(self, x, src_padding_mask=None, return_attn=False):
        # x shape: [batches, seq_len, embed_dim]
        # src_padding_mask shape: [batches, seq_len]
        b, s, e = x.shape

        x = torch.cat([repeat(self.groups, 'g d -> b g d', b=b), x], dim=1)

        if src_padding_mask is not None:
            padding_mask_w_grp = torch.cat([
                torch.zeros((b, self.num_groups), dtype=torch.bool, device=src_padding_mask.device),
                src_padding_mask,
            ], dim=1)
        else:
            padding_mask_w_grp = None

        x = self.transformer_layers(x, src_key_padding_mask=padding_mask_w_grp)
        return self.grouping_layer(x, src_padding_mask, return_attn)


class EmbeddingProjection(nn.Module):
    """
    Project the melspectrogram features to embed_dim dimensions

    MIGHT WANT TO EXPERIMENT WITH CONV1D HERE INSTEAD OF LINEAR
    """
    def __init__(self, n_features, embed_dim):
        super().__init__()
        self.embed = nn.Linear(n_features, embed_dim)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.pos = nn.Parameter(torch.rand((max_seq_len, embed_dim)))

    def forward(self, x):
        return x + self.pos[:x.shape[1]]


class AudioSegmenter(nn.Module):
    """
    An audio segmenter in the spirit of GroupViT. Consists of a number of grouping
    stages, followed by a stack of transformer encoder layers. A final projection
    layer transforms the group tokens into an audio-text embedding space. While
    training, we take the average of the output tokens before doing the projection.
    During inference, we take projections of all the output group tokens.

    Initialization:
        - n_features: Number of features in the input (for embedding projection)
        - max_seq_len: Maximum length of the input sequence (for positional embedding)
        - num_grouping_stages: Number of grouping stages
        - groups_per_stage: Number of group tokens in each grouping stage
        - layers_per_stage: Number of transformer encoder layers in each grouping stage
        - final_attn_layers: Number of encoder layers after grouping stages
        - final_proj_dim: Size of the final audio-text projection
        - encoder_path: if not None, path from which to load finetuned AudioSegmenter
        - other params: Same as Grouping Stage

    Inputs:
        - x: [batches, seq_tokens, n_features]
        - src_padding_mask: None, or [batches, seq_len] boolean tensor. False positions
            not a padding token, and True positions are padding to be masked.
        - return_attn: Whether to return the attention weights.
        - avg: Whether to average before projection (for training)

    Outputs:
        - out: [batches, 1 if avg else final_groups, final_proj_dim]
    """
    def __init__(
        self,
        n_features: int = 128,
        max_seq_len: int = 1000,
        num_grouping_stages: int = 2,
        groups_per_stage: list[int] = [64, 16],
        hard: bool = True,
        gumbel: bool = True,
        gumbel_tau: float = 1.,
        layers_per_stage: list[int] = [6, 3],
        final_attn_layers: int = 3,
        embed_dim: int = 384,
        mlp_scale: int = 4,
        num_heads: int = 8,
        transformer_activation: str = "relu",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
        final_proj_dim: int = 512,
        encoder_path: str = None,
        last_epoch: int = None
    ):
        super().__init__()

        self.num_grouping_stages = num_grouping_stages

        self.embed = EmbeddingProjection(n_features, embed_dim)
        self.pos_enc = PositionalEncoding(max_seq_len, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.stages = nn.ModuleList()

        for stage_idx in range(num_grouping_stages):
            grouping_stage = GroupingStage(
                num_groups=groups_per_stage[stage_idx],
                embed_dim=embed_dim,
                gumbel=gumbel,
                gumbel_tau=gumbel_tau,
                hard=hard,
                num_layers=layers_per_stage[stage_idx],
                num_heads=num_heads,
                mlp_scale=mlp_scale,
                transformer_activation=transformer_activation,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout,
            )

            self.stages.append(grouping_stage)

        self.final_attn_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_scale,
                dropout=dropout,
                activation=transformer_activation,
                batch_first=True,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=final_attn_layers,
        )

        self.head = nn.Linear(embed_dim, final_proj_dim)

        if encoder_path is not None:
            if last_epoch is None:
                raise ValueError("LAST_EPOCH: Need an epoch number to load model")
            
            print("Initializing audio segmenter from local finetuned model...")
            self.load_state_dict(torch.load(Path(encoder_path) / f'segmenter_epoch_{last_epoch}.pth'))

    def forward(self, x, src_padding_mask=None, return_attn=False, avg=True):
        attn_dicts = []

        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.embed_norm(x)

        # Padding mask is only needed for the first grouping stage
        x, attn_dict = self.stages[0](x, src_padding_mask, return_attn)
        attn_dicts.append(attn_dict)

        for i in range(1, self.num_grouping_stages):
            x, attn_dict = self.stages[i](x, return_attn=return_attn)
            attn_dicts.append(attn_dict)

        x = self.final_attn_block(x)

        if avg:
            x = x.mean(dim=1, keepdim=True)

        return self.head(x), attn_dicts

    def save_model(self, save_path: str, epoch_end: int):
        torch.save(self.state_dict(), Path(save_path) / f'segmenter_epoch_{epoch_end}.pth')