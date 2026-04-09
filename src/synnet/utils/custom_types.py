"""Definition of custom types for type hinting"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

PathType = str | Path
MetricType = (
    str | Callable[[npt.NDArray[Any], npt.NDArray[Any]], npt.NDArray[np.float64]]
)
