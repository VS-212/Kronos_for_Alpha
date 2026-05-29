"""
Setup HuggingFace token for gated Kronos models.

Kronos models (Kronos-mini, Kronos-small, Kronos-Tokenizer-2k)
are gated on HuggingFace Hub. HF_TOKEN is REQUIRED to download them.

Steps:
1. Get token: https://huggingface.co/settings/tokens → Create "read" token
2. Run this script: python scripts/setup_hf_token.py
   OR set environment variable: export HF_TOKEN=hf_xxxxxxxxx

The token persists in ~/.cache/huggingface/token for HF Hub library.
"""

import os
from pathlib import Path

HF_CACHE_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
MODELS = [
    "NeoQuasar/Kronos-mini",
    "NeoQuasar/Kronos-small",
    "NeoQuasar/Kronos-Tokenizer-2k",
]


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        HF_CACHE_TOKEN.parent.mkdir(parents=True, exist_ok=True)
        HF_CACHE_TOKEN.write_text(token)
        print(f"HF_TOKEN set via env var → {HF_CACHE_TOKEN}")
        print("Token cached for huggingface_hub library.")
        return

    if HF_CACHE_TOKEN.exists():
        token = HF_CACHE_TOKEN.read_text().strip()
        print(f"Token found at {HF_CACHE_TOKEN}")
        if token.startswith("hf_"):
            print(f"Token ok ({token[:10]}...{token[-4:]})")
        else:
            print("WARNING: token doesn't start with 'hf_'")
            print("Get a token: https://huggingface.co/settings/tokens")
        return

    print()
    print("=" * 60)
    print("HF_TOKEN NOT FOUND")
    print("=" * 60)
    print()
    print("These gated models require HuggingFace authentication:")
    for m in MODELS:
        print(f"  • {m}")
    print()
    print("To get a token:")
    print("  1. Go to https://huggingface.co/settings/tokens")
    print("  2. Create a new 'read' token")
    print("  3. Run this script again:")
    print()
    print("     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx")
    print("     python scripts/setup_hf_token.py")
    print()
    print("  Or set it per-session:")
    print("     HF_TOKEN=hf_xxx python -m src.core.kronos.predictor ...")
    print()
    print("For Modal deployment:")
    print("  modal secret create huggingface HF_TOKEN=hf_xxx")
    print()


if __name__ == "__main__":
    main()
