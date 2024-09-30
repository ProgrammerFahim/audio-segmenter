PROJECT_NAME = "audio_segmentation_train_100"

EPOCHS = 2
LAST_EPOCH = None
VERSION = None # None, or a version number like "1" or "2"
BATCH_SIZE = 8
FINETUNING = False
LOG_EVERY_N_STEPS = 16
COSINE_SCHED_T_MAX = 15
GRAD_ACCUMULATION_STEPS = 8

TEXT_ENCODER = "bert"
TEXT_ENCODER_SAVE_PATH = f'trained_models/{TEXT_ENCODER}/text'
AUDIO_ENCODER_SAVE_PATH = f'trained_models/{TEXT_ENCODER}/audio'
UTILS_SAVE_PATH = f'trained_models/{TEXT_ENCODER}'

WANDB_API_KEY = "INSERT WANDB API KEY HERE" 
