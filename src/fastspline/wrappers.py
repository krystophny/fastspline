"""High-level wrappers for bisplrep/bisplev with SciPy-compatible interface."""

import numpy as np
from .bisplev_fast import bisplev_fast as bisplev
from .bisplrep_dierckx import bisplrep_dierckx as bisplrep

# bisplev is already defined in bisplev_dierckx.py with full SciPy compatibility
# Just re-export it here

__all__ = ['bisplrep', 'bisplev']