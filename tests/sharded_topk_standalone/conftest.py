"""Standalone tests require only torch + pytest, not transformer_lens/Megatron.

Run this directory alone with --confcutdir=tests/sharded_topk_standalone.
Never interpret these dependency-light tests as native Megatron/CUDA acceptance.
"""

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
# An isolated namespace lets the new dependency-light modules import each other
# without executing SAELens' LLM-oriented package initializer.
if "sae_lens" not in sys.modules:
    pkg = types.ModuleType("sae_lens")
    pkg.__path__ = [str(ROOT / "sae_lens")]
    sys.modules["sae_lens"] = pkg
torch.set_num_threads(1)
