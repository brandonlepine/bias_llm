"""Default model resolution.

Returns the LOCAL model dir if it exists (so a machine that already has the
weights, e.g. the dev laptop, never re-downloads), otherwise the HuggingFace
repo id so the model is downloaded on demand (e.g. on a fresh pod). On the pod
just authenticate first (`huggingface-cli login` or export HF_TOKEN); transformers
then pulls and caches the weights automatically.
"""
import os

HF_REPO_ID = "meta-llama/Llama-3.1-8B"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCAL_MODEL = os.path.join(_REPO_ROOT, "models", "Llama-3.1-8B")


def default_model():
    """Local weights dir if present, else the HF repo id (auto-download)."""
    return _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else HF_REPO_ID


def dtype_kwargs(dtype):
    """transformers >=5 uses `dtype=`, <5 uses `torch_dtype=` (passing the wrong
    one on the other version silently loads fp32). Pick the right kwarg."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
