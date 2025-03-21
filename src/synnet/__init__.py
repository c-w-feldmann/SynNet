"""Initialize the synnet package."""

import os

RUNNING_ON_HPC: bool = "SLURM_JOB_ID" in os.environ
