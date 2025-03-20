from pathlib import Path
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base paths
SRC_DIR = Path(__file__).parent  # Gets the directory where config.py is
DATA_DIR = SRC_DIR / "data"      # data directory inside src
WDS_DIR = SRC_DIR / "data" / "wds"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_FREQUENCY = 1000
METADATA_EXTENSION = ".json"

# DB paths
DB_PATH = DATA_DIR / "images.db"
INDEX_PATH = DATA_DIR / "embeddings.ann"

# Model config
MODEL_NAME = 'ViT-B-32'
PRETRAINED = "laion2b_s34b_b79k" #previous. used "datacomp_s_s13m_b4k" to test the pipeline real quick.Now set max_samples instead.
BATCH_SIZE = 16
MAX_SAMPLES = 5000
EPOCHS = 10
ETA = 5e-5 
VECTOR_DIM = 512
IMAGE_SIZE = 224


# Dataset config
LANGUAGES = 'eng'

# Wandb config
WANDB_PROJECT = "my-open-clip"

# Training config
N_TREES = 100  # For Annoy index building

# Evaluation config
EVAL_BATCH_SIZE = 32
TOP_K = [1, 5, 10]  # For retrieval evaluation
SIMILARITY_THRESHOLD = 0.8

# directories creation
for dir_path in [DATA_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


