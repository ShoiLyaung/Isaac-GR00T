from .gr00t_n1d6.setup import Gr00tN1d6Pipeline
from .registry import MODEL_REGISTRY

# N1.7 vendored files. Guarded import: if N1.7 deps (transformers >= 4.57,
# Qwen3VLProcessor) are missing, the N1.6 path is unaffected.
# Importing ``gr00t_n1d7.setup`` triggers the
# ``register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)`` side effect that
# populates MODEL_REGISTRY.
try:
    from .gr00t_n1d7.setup import (  # noqa: F401
        Gr00tN1d7Pipeline,
    )

    _N1D7_AVAILABLE = True
except ImportError:
    _N1D7_AVAILABLE = False
