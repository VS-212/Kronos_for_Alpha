"""
Modal job for Kronos inference on GPU (T4 / A100).

Pattern: PR_Kronos/alpha/infra/jobs.py
  - Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
  - Volumes: HF cache + predictions output
  - Functions: seed (download models), infer_10min, infer_1hour

Usage:
  modal run scripts/modal_inference.py::seed           # one-time: cache models
  modal run scripts/modal_inference.py::infer_10min    # SBER 10-min (T4)
  modal run --detach scripts/modal_inference.py::infer_10min  # background
  modal logs --tail 100 kronos-inference               # check logs
  modal volume get kronos-predictions /path/ ./        # download results
"""

import sys
from pathlib import Path

import modal

_repo_root = Path(__file__).resolve().parent.parent

DEPS = [
    "apimoex",
    "pandas", "numpy", "pyarrow",
    "safetensors", "pyyaml",
    "huggingface_hub", "requests",
    "einops", "tqdm", "scipy",
]

KronosImage = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
    .pip_install(*DEPS)
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "PYTHONPATH": "/root",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(str(_repo_root / "src"), remote_path="/root/src")
    .add_local_dir(str(_repo_root / "config"), remote_path="/root/config")
    .add_local_dir(str(_repo_root / "data"), remote_path="/root/data")
)

hf_vol = modal.Volume.from_name("kronos-hf-cache", create_if_missing=True)
pred_vol = modal.Volume.from_name("kronos-predictions", create_if_missing=True)

app = modal.App("kronos-inference", image=KronosImage)


@app.function(
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_vol},
    timeout=600,
)
def seed():
    """One-time: download model weights to HF cache volume."""
    from huggingface_hub import snapshot_download
    repos = [
        "NeoQuasar/Kronos-mini",
        "NeoQuasar/Kronos-small",
        "NeoQuasar/Kronos-Tokenizer-2k",
        "NeoQuasar/Kronos-Tokenizer-base",
    ]
    for repo in repos:
        print(f"Downloading {repo}...", flush=True)
        snapshot_download(repo, ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
    hf_vol.commit()
    print("Done. Models cached.", flush=True)


def _run_inference(
    model_name: str = "NeoQuasar/Kronos-mini",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
    pred_len: int = 12,
    lookback: int = 500,
    sample_count: int = 5,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed_val: int = 42,
    sub_batch: int = 8,
    use_bf16: bool = True,
    data_path: str = "/root/data/combined_dataset.parquet",
    ticker_cols: tuple = ("SBER_open", "SBER_high", "SBER_low", "SBER_close", "SBER_volume"),
    test_start: str = "2025-11-20",
    output_dir: str = "/root/predictions",
    tag: str = "10min",
):
    """Run inference with belief extraction on raw OHLCV data."""
    import gc

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root")
    from src.core.kronos.predictor import KronosModel

    # ── Load model ──
    km = KronosModel(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        seed=seed_val,
        use_bf16=use_bf16,
    )
    km.load()
    km.predictor.max_context = lookback + pred_len
    print(f"Model loaded: {model_name}", flush=True)

    # ── Load and prepare data ──
    df = pd.read_parquet(data_path)
    ts = pd.to_datetime(df["timestamp"])
    cols_list = list(ticker_cols)
    ticker_df = df[cols_list].copy()
    ticker_df.columns = ["open", "high", "low", "close", "volume"]
    ticker_df.index = pd.DatetimeIndex(ts)
    ticker_df["amount"] = ticker_df["close"] * ticker_df["volume"]

    # Session filter
    session = (ticker_df.index.hour * 60 + ticker_df.index.minute >= 600) & (
        ticker_df.index.hour * 60 + ticker_df.index.minute <= 1120
    )
    ticker_df = ticker_df[session].copy()

    # Test split
    test_mask = ticker_df.index >= test_start
    df_test = ticker_df[test_mask].copy()
    n_total = len(df_test)
    n_windows = n_total - lookback - pred_len + 1
    print(f"Data: {n_total} bars, {n_windows} windows", flush=True)

    # ── Batch inference ──
    all_preds = []
    all_beliefs = []

    for batch_start in range(0, n_windows, sub_batch):
        batch_end = min(batch_start + sub_batch, n_windows)
        batch_ctx = []
        for w in range(batch_start, batch_end):
            t = lookback + w
            ctx = df_test.iloc[t - lookback : t].copy()
            batch_ctx.append(ctx)

        results, beliefs = km.predict_samples_batch(
            batch_ctx,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            return_beliefs=True,
        )
        all_preds.extend(results)
        all_beliefs.extend(beliefs)

        if batch_end % 100 == 0 or batch_end == n_windows:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [{batch_end}/{n_windows}] {batch_end/n_windows*100:.0f}%", flush=True)

    # ── Save ──
    out = Path(output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"SBER_preds_pl{pred_len}_sc{sample_count}.npy", np.stack(all_preds, axis=0))
    np.save(out / f"SBER_belief_pl{pred_len}_sc{sample_count}.npy", np.stack(all_beliefs, axis=0))
    pred_vol.commit()
    print(f"Done. {n_windows} windows → {out}", flush=True)


@app.function(
    gpu="T4",
    memory=16384,
    timeout=3600,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/root/predictions": pred_vol},
)
def infer_10min():
    """10-min SBER mini, pred_len=12, MC=5, belief=True on T4."""
    _run_inference(
        model_name="NeoQuasar/Kronos-mini",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
        pred_len=12, lookback=500, sample_count=5,
        temperature=0.6, top_p=0.9, seed_val=42,
        sub_batch=8, use_bf16=True,
        tag="10min_sber",
    )


@app.function(
    gpu="T4",
    memory=16384,
    timeout=3600,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/root/predictions": pred_vol},
)
def infer_10min_small():
    """10-min SBER small, pred_len=12, MC=5, belief=True on T4."""
    _run_inference(
        model_name="NeoQuasar/Kronos-small",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        pred_len=12, lookback=500, sample_count=5,
        temperature=0.6, top_p=0.9, seed_val=42,
        sub_batch=8, use_bf16=True,
        tag="10min_sber_small",
    )


@app.function(
    gpu="T4",
    memory=16384,
    timeout=3600,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/root/predictions": pred_vol},
)
def infer_1hour():
    """1-hour SBER, pred_len=2, MC=5, belief=True on T4."""
    _run_inference(
        pred_len=2, lookback=510, sample_count=5,
        temperature=0.6, top_p=0.9, seed_val=42,
        sub_batch=8, use_bf16=True,
        data_path="/root/data/v3/1h/raw/SBER.parquet",
        ticker_cols=("open", "high", "low", "close", "volume"),
        test_start="2025-11-20",
        tag="1hour_sber",
    )


@app.function(
    gpu="A100",
    memory=40960,
    timeout=1800,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/root/predictions": pred_vol},
)
def infer_10min_a100():
    """10-min SBER mini, lookback=2036, pred_len=12 on A100."""
    _run_inference(
        model_name="NeoQuasar/Kronos-mini",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
        pred_len=12, lookback=2036, sample_count=5,
        temperature=0.6, top_p=0.9, seed_val=42,
        sub_batch=16, use_bf16=True,
        tag="10min_sber_mini_l2036",
    )


@app.function(
    gpu="A100",
    memory=40960,
    timeout=1800,
    retries=0,
    volumes={"/root/.cache/huggingface": hf_vol, "/root/predictions": pred_vol},
)
def infer_10min_small_a100():
    """10-min SBER small on A100 (faster, larger sub_batch)."""
    _run_inference(
        model_name="NeoQuasar/Kronos-small",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        pred_len=12, lookback=500, sample_count=5,
        temperature=0.6, top_p=0.9, seed_val=42,
        sub_batch=16, use_bf16=True,
        tag="10min_sber_small",
    )
