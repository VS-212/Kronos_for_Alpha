"""
M-REGISTRY: Model registry and base model interface
Contract: register / get / list models; BaseModel ABC for prediction API
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with columns ['open','high','low','close'] and datetime index.
                Must contain lookback candles (up to max_context). Volume/amount optional.
            pred_len: Number of future candles to predict.
            T: Sampling temperature.
            top_p: Nucleus sampling threshold.
            sample_count: Number of Monte Carlo paths (averaged internally).
        Returns:
            DataFrame with ['open','high','low','close','volume','amount']
            indexed by future timestamps.
        """
        ...

    @abstractmethod
    def load(self):
        """Load model and tokenizer from HF Hub."""
        ...

    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> list[pd.DataFrame]:
        """Default batch: loop over predict(). Override for true parallelism."""
        return [
            self.predict(df, pred_len, T=T, top_p=top_p, sample_count=sample_count)
            for df in df_list
        ]

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# ── Model registry ─────────────────────────────────────────────────────

_model_registry = {}


def register_model(name: str, model_class):
    _model_registry[name] = model_class


def get_model(name: str, **kwargs):
    if name not in _model_registry:
        raise KeyError(
            f"Model '{name}' not registered. Available: {sorted(_model_registry.keys())}"
        )
    return _model_registry[name](**kwargs)


def list_models():
    return list(_model_registry.keys())


# ── Default registrations (lazy import avoids circular dependency) ─────

from src.core.kronos.predictor import KronosModel

register_model("kronos_mini", KronosModel)
register_model(
    "kronos_mini_2048",
    lambda **kw: KronosModel(
        model_name="NeoQuasar/Kronos-mini",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        **kw,
    ),
)
