# If dependencies are missing, install them in your environment first.
# Colab normally includes these packages.
from pathlib import Path
import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = Path("data")
# Mount Drive in Colab; fail visibly rather than silently saving to temporary storage.
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = Path('/content/drive/MyDrive/EN4554_220381M/outputs')
else:
    OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = OUTPUT_DIR / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)
print('Persistent results/checkpoints directory:', OUTPUT_DIR)
BATCH_SIZE = 128
NUM_WORKERS = 0  # Portable default for Windows notebooks and Colab.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))

# Atomic replacement leaves the previous checkpoint intact if writing is interrupted.
def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)

def atomic_json_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)

def capture_rng(loader):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader': loader.generator.get_state()}

def restore_rng(state, loader):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        if len(state['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    loader.generator.set_state(state['loader'].cpu())

# Every displayed figure also gets a PNG backup on Drive.
import time as _backup_time
if not hasattr(plt, '_assignment_original_show'):
    plt._assignment_original_show = plt.show

def show_and_save(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figure.savefig(FIGURE_DIR / f'figure_{_backup_time.time_ns()}_{number}.png',
                       dpi=150, bbox_inches='tight')
    return plt._assignment_original_show(*args, **kwargs)

plt.show = show_and_save
