from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'
RANDOM_SEED=42

SEQ_LEN=5
BATCH_SIZE=128
EMBEDDING_DIM=128
HIDDEN_SIZE=256
LEARNING_RATE=0.001
EPOCHS=10