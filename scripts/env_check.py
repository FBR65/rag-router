"""P1-Pruefung: alle Kernimporte laden und Versionszahlen melden."""

import lancedb
import numpy
import openai
import scipy
import torch
import transformers
import yaml


def main() -> None:
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
    print("lancedb", lancedb.__version__)
    print("transformers", transformers.__version__)
    print("numpy", numpy.__version__, "| scipy", scipy.__version__)
    print("pyyaml", yaml.__version__, "| openai", openai.__version__)
    print("rank_bm25 ok", "| FlagEmbedding ok")
    assert not torch.cuda.is_available(), "CPU-Build erwartet"


if __name__ == "__main__":
    main()
