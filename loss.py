import torch
import torch.nn.functional as F

def compute_loss(audio_proj, text_proj, word_mat):
    """ Compute Audio and Text Loss """
    batch = audio_proj.shape[0]

    audio_text_logits = torch.matmul(audio_proj, text_proj[:batch].T)               # [batch, batch]
    labels = torch.arange(len(audio_text_logits), device=audio_text_logits.device)  # [batch,]

    audio_loss = F.cross_entropy(audio_text_logits, labels)
    text_loss = F.cross_entropy(audio_text_logits.T, labels)

    general_loss = audio_loss + text_loss

    """ Compute Vocab Loss """
    audio_vocab_logits = torch.matmul(audio_proj, text_proj[batch:].T)              # [batch, batch_vocab]

    vocab_loss = F.binary_cross_entropy_with_logits(audio_vocab_logits, word_mat)

    """ Total loss """
    total_loss = general_loss + vocab_loss

    return total_loss, (audio_loss, text_loss, vocab_loss)