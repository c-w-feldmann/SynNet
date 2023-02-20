"""Central place for all configuration, paths, and parameter."""
import multiprocessing
import os

# Multiprocessing
DEFAULT_MAX_PROCESSES = 31
MAX_PROCESSES = min(
    int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_MAX_PROCESSES)), multiprocessing.cpu_count()
)


# Default split ratio train/val/test
SPLIT_RATIO = [0.6, 0.2, 0.2]

# Number of reaction templates
# (used for 1-hot-encoding of reaction templates)
DEFAULT_NUM_RXN_TEMPLATES = 91
NUM_RXN_TEMPLATES = os.environ.get("SYNNET_NUM_RXN_TEMPLATES", DEFAULT_NUM_RXN_TEMPLATES)
