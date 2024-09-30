from .config import TEXT_ENCODER

ENCODER_NAME = "bert-base-uncased" if TEXT_ENCODER == "bert" else "distilbert-base-uncased"
POOLER = True if TEXT_ENCODER == "bert" else False