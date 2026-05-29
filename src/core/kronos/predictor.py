"""
M-PREDICT: Kronos inference and prediction wrappers
Contract: tokenizer + model → autoregressive sampling → OHLCV predictions [B,pred_len,6]
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
    sample_logits=True,
    generator=None,
):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits.float(), dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1, generator=generator)

    return x


def auto_regressive_inference(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
    seed=None,
    use_bf16=False,
    return_beliefs=False,
):
    """Autoregressive inference with averaging across sample paths.

    Args:
        seed: If set, makes sampling deterministic (per-run reproducibility).
        use_bf16: If True, run model forward in bfloat16 (faster on A100/T4).
        return_beliefs: If True, also return per-step belief metrics.

    Returns:
        np.ndarray or tuple: (preds, beliefs) if return_beliefs else preds.
        preds: [B, pred_len, 6] MC-averaged OHLCV.
        beliefs: [B, sample_count, pred_len, 4] — confidence, entropy_s1, top3_mass, entropy_ratio.
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        device = x.device

        _B_orig = x.size(0)
        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        if verbose:
            ran = trange
        else:
            ran = range

        # Belief accumulation
        belief_list = [] if return_beliefs else None

        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                s1_logits, context = model.decode_s1(
                    input_tokens[0], input_tokens[1], current_stamp
                )
            s1_logits = s1_logits[:, -1, :]

            # ── Belief extraction from s1_logits ──
            if return_beliefs:
                prob_s1 = F.softmax(s1_logits.float(), dim=-1)
                conf_s1 = prob_s1.max(-1).values
                H_s1 = -(prob_s1 * prob_s1.log2()).sum(-1)
                top3_s1 = prob_s1.topk(3).values.sum(-1)

            # Deterministic per-call seed: batch-size-independent, MC-diverse
            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 0)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + i * 1000 + 0)

            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]

            # ── Belief extraction from s2_logits ──
            if return_beliefs:
                prob_s2 = F.softmax(s2_logits.float(), dim=-1)
                H_s2 = -(prob_s2 * prob_s2.log2()).sum(-1)
                entropy_ratio = H_s2 / (H_s1 + 1e-8)
                step_belief = torch.stack(
                    [conf_s1, H_s1, top3_s1, entropy_ratio], dim=-1
                )  # (batch_size, 4)
                belief_list.append(step_belief)

            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 1)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + i * 1000 + 1)

            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        # Assemble beliefs
        if return_beliefs:
            # belief_list: list of (pred_len,) tensors, each (batch_size, 4)
            beliefs = torch.stack(belief_list, dim=1)  # (batch_size, pred_len, 4)
            beliefs = beliefs.reshape(-1, sample_count, pred_len, 4)
            beliefs = beliefs.cpu().numpy()
            return preds, beliefs

        return preds


def auto_regressive_inference_raw(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
    seed=None,
    use_bf16=False,
    return_beliefs=False,
):
    """Autoregressive inference returning per-sample paths (no averaging).

    Args:
        seed: If set, makes sampling deterministic.
        use_bf16: If True, run model forward in bfloat16.
        return_beliefs: If True, also return per-step belief metrics.

    Returns:
        np.ndarray or tuple: (preds, beliefs) if return_beliefs else preds.
        preds: [B, sample_count, total_seq_len, 6] raw MC paths.
        beliefs: [B, sample_count, pred_len, 4] — confidence, entropy_s1, top3_mass, entropy_ratio.
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

        _B_orig = x.size(0)
        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        belief_list = [] if return_beliefs else None

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                s1_logits, context = model.decode_s1(
                    input_tokens[0], input_tokens[1], current_stamp
                )
            s1_logits = s1_logits[:, -1, :]

            if return_beliefs:
                prob_s1 = F.softmax(s1_logits.float(), dim=-1)
                conf_s1 = prob_s1.max(-1).values
                H_s1 = -(prob_s1 * prob_s1.log2()).sum(-1)
                top3_s1 = prob_s1.topk(3).values.sum(-1)

            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 0)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + i * 1000 + 0)

            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]

            if return_beliefs:
                prob_s2 = F.softmax(s2_logits.float(), dim=-1)
                H_s2 = -(prob_s2 * prob_s2.log2()).sum(-1)
                entropy_ratio = H_s2 / (H_s1 + 1e-8)
                step_belief = torch.stack(
                    [conf_s1, H_s1, top3_s1, entropy_ratio], dim=-1
                )
                belief_list.append(step_belief)

            if seed is not None:
                torch.manual_seed(seed + i * 1000 + 1)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + i * 1000 + 1)
            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()

        if return_beliefs:
            beliefs = torch.stack(belief_list, dim=1)
            beliefs = beliefs.reshape(-1, sample_count, pred_len, 4)
            beliefs = beliefs.cpu().numpy()
            return preds, beliefs

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df["minute"] = x_timestamp.dt.minute
    time_df["hour"] = x_timestamp.dt.hour
    time_df["weekday"] = x_timestamp.dt.weekday
    time_df["day"] = x_timestamp.dt.day
    time_df["month"] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        max_context=512,
        clip=5,
        seed=None,
        use_bf16=False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.seed = seed
        self.use_bf16 = use_bf16
        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_vol = "amount"
        self.time_cols = ["minute", "hour", "weekday", "day", "month"]

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # GPU optimizations
        if self.device and "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(
        self,
        x,
        x_stamp,
        y_stamp,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
        return_beliefs=False,
    ):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        result = auto_regressive_inference(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
            seed=self.seed,
            use_bf16=self.use_bf16,
            return_beliefs=return_beliefs,
        )
        if return_beliefs:
            preds, beliefs = result
            preds = preds[:, -pred_len:, :]
            return preds, beliefs
        preds = result
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # Fill missing volume with zeros
            df[self.amt_vol] = 0.0  # Fill missing amount with zeros
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(
            preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp
        )
        return pred_df

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
        return_beliefs=False,
    ):
        """
        Perform parallel (batch) prediction on multiple time series. All series must have the same historical length and prediction length (pred_len).

        Args:
            df_list (list[pd.DataFrame]): Input DataFrames with price columns.
            x_timestamp_list (list like): Historical timestamps per series.
            y_timestamp_list (list like): Future timestamps per series, length == pred_len.
            pred_len (int): Prediction horizon.
            T (float): Sampling temperature.
            top_k (int): Top-k filtering threshold.
            top_p (float): Top-p (nucleus sampling) threshold.
            sample_count (int): MC samples per series.
            verbose (bool): Show progress.
            return_beliefs (bool): If True, also return belief dicts.

        Returns:
            list[pd.DataFrame] or tuple: (pred_dfs, belief_dfs) if return_beliefs.
        """
        if (
            not isinstance(df_list, (list, tuple))
            or not isinstance(x_timestamp_list, (list, tuple))
            or not isinstance(y_timestamp_list, (list, tuple))
        ):
            raise ValueError(
                "df_list, x_timestamp_list, y_timestamp_list must be list or tuple types."
            )
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError(
                "df_list, x_timestamp_list, y_timestamp_list must have consistent lengths."
            )

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(
                    f"DataFrame at index {i} is missing price columns {self.price_cols}."
                )

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(
                    f"DataFrame at index {i} contains NaN values in price or volume columns."
                )

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(
                    f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}."
                )
            if y_stamp.shape[0] != pred_len:
                raise ValueError(
                    f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}."
                )

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(
                f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}"
            )
        if len(set(y_lens)) != 1:
            raise ValueError(
                f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}"
            )

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        result = self.generate(
            x_batch,
            x_stamp_batch,
            y_stamp_batch,
            pred_len,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
            return_beliefs=return_beliefs,
        )
        if return_beliefs:
            preds, beliefs = result
        else:
            preds = result

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(
                preds_i,
                columns=self.price_cols + [self.vol_col, self.amt_vol],
                index=y_timestamp_list[i],
            )
            pred_dfs.append(pred_df)

        if return_beliefs:
            # beliefs shape: (B, sample_count, pred_len, 4) — raw numpy
            return pred_dfs, beliefs

        return pred_dfs


# ── KronosModel: high-level model wrapper ─────────────────────────────
# Implements the BaseModel interface (predict, load, predict_batch, __call__).
# Does NOT formally inherit from BaseModel to avoid circular imports with
# src.core.registry (which imports KronosModel). The registry stores class
# references only; ABC registration is handled by duck typing.


class KronosModel:
    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-mini",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
        device: str | None = None,
        max_context: int = 2048,
        session_filter: bool = True,
        main_session_start: str = "10:00",
        main_session_end: str = "18:40",
        freq: str = "5min",
        seed: int | None = None,
        use_bf16: bool = False,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.max_context = max_context
        self.session_filter = session_filter
        self.main_session_start = main_session_start
        self.main_session_end = main_session_end
        self.freq = freq
        self.seed = seed
        self.use_bf16 = use_bf16
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.predictor = None

    def load(self):
        from src.core.kronos.model import Kronos
        from src.core.kronos.tokenizer import KronosTokenizer

        self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        self.model = Kronos.from_pretrained(self.model_name)
        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=self.device,
            max_context=self.max_context,
            seed=self.seed,
            use_bf16=self.use_bf16,
        )
        self._loaded = True

    def _filter_session(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.session_filter:
            return df
        start_t = pd.Timestamp(self.main_session_start).time()
        end_t = pd.Timestamp(self.main_session_end).time()
        mask = (df.index.time >= start_t) & (df.index.time <= end_t)
        return df[mask]

    def _validate_df(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}"
            )
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. Found: {sorted(df.columns)}"
            )
        if len(df) == 0:
            raise ValueError("DataFrame is empty after session filtering")

    def _resolve_freq(self, df: pd.DataFrame) -> str:
        inferred = pd.infer_freq(df.index)
        if inferred is not None:
            return inferred
        if len(df) >= 2:
            inferred = pd.infer_freq(df.index[: min(100, len(df))])
            if inferred is not None:
                return inferred
        return self.freq

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        samples = self.predict_samples(df, pred_len, T, 0, top_p, sample_count)
        mean_preds = np.mean(samples, axis=0)
        return pd.DataFrame(
            mean_preds,
            columns=self.predictor.price_cols + [self.predictor.vol_col, self.predictor.amt_vol],
        )

    def predict_samples(
        self,
        df: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 5,
        return_beliefs: bool = False,
    ) -> np.ndarray | tuple:
        if not self._loaded:
            self.load()

        df = self._filter_session(df).copy()
        self._validate_df(df)

        price_cols = self.predictor.price_cols
        vol_col = self.predictor.vol_col
        amt_col = self.predictor.amt_vol
        clip = self.predictor.clip

        if vol_col not in df.columns:
            df[vol_col] = 0.0
            df[amt_col] = 0.0
        if amt_col not in df.columns and vol_col in df.columns:
            df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)

        x_timestamp = df.index.to_series()
        last_ts = df.index[-1]
        freq = self._resolve_freq(df)

        y_timestamp = pd.date_range(
            start=last_ts + pd.Timedelta(freq),
            periods=pred_len,
            freq=freq,
            tz=last_ts.tz if hasattr(last_ts, "tz") else None,
        )

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(pd.Series(y_timestamp))

        x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -clip, clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        x_tensor = torch.from_numpy(x).to(self.predictor.device)
        x_stamp_tensor = torch.from_numpy(x_stamp).to(self.predictor.device)
        y_stamp_tensor = torch.from_numpy(y_stamp).to(self.predictor.device)

        result = auto_regressive_inference_raw(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose=False,
            seed=self.seed,
            use_bf16=self.use_bf16,
            return_beliefs=return_beliefs,
        )
        if return_beliefs:
            preds_all, beliefs = result
            preds_all = preds_all[:, :, -pred_len:, :]
            preds_all = preds_all * (x_std + 1e-5) + x_mean
            return preds_all.squeeze(0), beliefs.squeeze(0)
        preds_all = result
        preds_all = preds_all[:, :, -pred_len:, :]
        preds_all = preds_all * (x_std + 1e-5) + x_mean
        return preds_all.squeeze(0)

    def predict_samples_batch(
        self,
        df_list: list[pd.DataFrame],
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 5,
        return_beliefs: bool = False,
    ) -> list[np.ndarray] | tuple:
        if not self._loaded:
            self.load()

        if not isinstance(df_list, (list, tuple)):
            raise ValueError("df_list must be a list or tuple of DataFrames")
        if len(df_list) == 0:
            raise ValueError("df_list is empty")

        price_cols = self.predictor.price_cols
        vol_col = self.predictor.vol_col
        amt_col = self.predictor.amt_vol
        clip = self.predictor.clip

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []

        for df in df_list:
            df_f = self._filter_session(df).copy()
            self._validate_df(df_f)

            if vol_col not in df_f.columns:
                df_f[vol_col] = 0.0
                df_f[amt_col] = 0.0
            if amt_col not in df_f.columns and vol_col in df_f.columns:
                df_f[amt_col] = df_f[vol_col] * df_f[price_cols].mean(axis=1)

            x_timestamp = df_f.index.to_series()
            last_ts = df_f.index[-1]
            freq = self._resolve_freq(df_f)
            y_timestamp = pd.date_range(
                start=last_ts + pd.Timedelta(freq),
                periods=pred_len,
                freq=freq,
                tz=last_ts.tz if hasattr(last_ts, "tz") else None,
            )

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(pd.Series(y_timestamp))

            x = df_f[price_cols + [vol_col, amt_col]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            seq_lens.append(x.shape[0])
            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -clip, clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

        if len(set(seq_lens)) != 1:
            raise ValueError(f"All series must have same historical length, got: {set(seq_lens)}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        x_tensor = torch.from_numpy(x_batch).to(self.predictor.device)
        x_stamp_tensor = torch.from_numpy(x_stamp_batch).to(self.predictor.device)
        y_stamp_tensor = torch.from_numpy(y_stamp_batch).to(self.predictor.device)

        result = auto_regressive_inference_raw(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose=False,
            seed=self.seed,
            use_bf16=self.use_bf16,
            return_beliefs=return_beliefs,
        )
        if return_beliefs:
            preds_all, beliefs_all = result
            # preds_all: (B, sample_count, pred_len, 6)
            # beliefs_all: (B, sample_count, pred_len, 4)
            preds_all = preds_all[:, :, -pred_len:, :]
            B = len(df_list)
            results = []
            beliefs_list = []
            for i in range(B):
                preds_i = preds_all[i] * (stds[i] + 1e-5) + means[i]
                results.append(preds_i)
                beliefs_list.append(beliefs_all[i])
            return results, beliefs_list

        preds_all = result
        preds_all = preds_all[:, :, -pred_len:, :]
        B = len(df_list)
        results = []
        for i in range(B):
            preds_i = preds_all[i] * (stds[i] + 1e-5) + means[i]
            results.append(preds_i)
        return results

    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        return_beliefs: bool = False,
    ) -> list[pd.DataFrame] | tuple:
        if not self._loaded:
            self.load()

        if not isinstance(df_list, (list, tuple)):
            raise ValueError("df_list must be a list or tuple of DataFrames")
        if len(df_list) == 0:
            raise ValueError("df_list is empty")

        filtered_dfs = []
        x_timestamp_list = []
        y_timestamp_list = []

        for df in df_list:
            df_f = self._filter_session(df).copy()
            self._validate_df(df_f)
            filtered_dfs.append(df_f)

            x_timestamp_list.append(df_f.index.to_series())

            last_ts = df_f.index[-1]
            freq = self._resolve_freq(df_f)

            y_ts = pd.date_range(
                start=last_ts + pd.Timedelta(freq),
                periods=pred_len,
                freq=freq,
                tz=last_ts.tz if hasattr(last_ts, "tz") else None,
            )
            y_timestamp_list.append(y_ts)

        return self.predictor.predict_batch(
            df_list=filtered_dfs,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
            return_beliefs=return_beliefs,
        )

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# ── CLI ─────────────────────────────────────────────────────────────────


def _parse_ticker_cols(feats: np.ndarray, timestamps: np.ndarray,
                        ticker_names: list[str] | None = None) -> dict:
    """Parse wide-format features (N_tickers × 5_OHLCV cols) into per-ticker dicts.
    If ticker_names provided, use them instead of generic T0, T1, ...
    """
    n_cols = feats.shape[1]
    n_tickers = n_cols // 5
    if ticker_names is None:
        ticker_names = [f"T{i}" for i in range(n_tickers)]

    result = {}
    for t_idx, name in enumerate(ticker_names[:n_tickers]):
        col_start = t_idx * 5
        df = pd.DataFrame(
            feats[:, col_start : col_start + 5],
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex(timestamps),
        )
        df["amount"] = df["close"] * df["volume"]
        result[name] = df
    return result


def load_model(
    model_name: str = "NeoQuasar/Kronos-mini",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
    device: str | None = None,
    seed: int | None = None,
    use_bf16: bool = False,
) -> KronosModel:
    model = KronosModel(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        max_context=2048,
        seed=seed,
        use_bf16=use_bf16,
    )
    model.load()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kronos Predictor — inference with belief state")
    parser.add_argument("--feats", type=str, required=True, help="Path to features .npy (OHLCV)")
    parser.add_argument("--timestamps", type=str, required=True, help="Path to timestamps .npy")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--ticker-names", type=str, nargs="*", help="Ticker names (optional)")
    parser.add_argument("--pred-len", type=int, default=12, help="Prediction horizon in bars")
    parser.add_argument("--sample-count", type=int, default=5, help="MC sample count")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--lookback", type=int, default=500, help="Context window bars")
    parser.add_argument("--sub-batch", type=int, default=8, help="Windows per batch (T4: 8, A100: 16)")
    parser.add_argument("--model", type=str, default="NeoQuasar/Kronos-mini", help="HF model")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-2k", help="HF tokenizer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 precision")
    parser.add_argument("--belief", action="store_true", help="Extract belief state")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    km = load_model(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
        seed=args.seed,
        use_bf16=args.bf16,
    )

    feats = np.load(args.feats).astype(np.float32)
    timestamps = np.load(args.timestamps, allow_pickle=True)
    ticker_data = _parse_ticker_cols(feats, timestamps, ticker_names=args.ticker_names)
    ticker_names = list(ticker_data.keys())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    if args.belief:
        belief_dir = out_dir / "belief"
        belief_dir.mkdir(exist_ok=True)

    max_context = min(2048, args.lookback + args.pred_len)
    km.predictor.max_context = max_context

    all_ts = pd.DatetimeIndex(timestamps)
    n_total = len(all_ts)
    lookback = args.lookback
    pred_len = args.pred_len
    n_windows = n_total - lookback - pred_len + 1
    sub_batch = args.sub_batch

    print(f"Model: {args.model}", flush=True)
    print(f"Tickers: {ticker_names}", flush=True)
    print(f"Windows: {n_windows}, sub_batch: {sub_batch}, batches: {n_windows // sub_batch + 1}", flush=True)
    print(f"pred_len={pred_len}, lookback={lookback}, MC={args.sample_count}", flush=True)
    print(f"seed={args.seed}, bf16={args.bf16}, belief={args.belief}", flush=True)
    print(flush=True)

    all_preds = {name: [] for name in ticker_names}
    all_beliefs = {name: [] for name in ticker_names} if args.belief else None

    for batch_start in range(0, n_windows, sub_batch):
        batch_end = min(batch_start + sub_batch, n_windows)
        batch_ctx = []

        for w in range(batch_start, batch_end):
            t = lookback + w
            ctx_df = ticker_data[ticker_names[0]].iloc[t - lookback : t].copy()
            batch_ctx.append(ctx_df)

        if args.belief:
            results, beliefs = km.predict_samples_batch(
                batch_ctx, pred_len=pred_len, T=args.temperature,
                top_p=args.top_p, sample_count=args.sample_count,
                return_beliefs=True,
            )
            for w_idx in range(len(results)):
                all_preds[ticker_names[0]].append(results[w_idx])
                all_beliefs[ticker_names[0]].append(beliefs[w_idx])
        else:
            results = km.predict_samples_batch(
                batch_ctx, pred_len=pred_len, T=args.temperature,
                top_p=args.top_p, sample_count=args.sample_count,
            )
            for w_idx in range(len(results)):
                all_preds[ticker_names[0]].append(results[w_idx])

        if (batch_end) % 100 == 0 or batch_end == n_windows:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [{batch_end}/{n_windows}] {batch_end/n_windows*100:.0f}%", flush=True)

    for name in ticker_names:
        arr = np.stack(all_preds[name], axis=0)
        np.save(pred_dir / f"{name}_preds_pl{pred_len}_sc{args.sample_count}.npy", arr)

    if args.belief:
        for name in ticker_names:
            arr = np.stack(all_beliefs[name], axis=0)
            np.save(belief_dir / f"{name}_belief_pl{pred_len}_sc{args.sample_count}.npy", arr)

    print(f"Done. Predictions: {pred_dir}", flush=True)
    if args.belief:
        print(f"Beliefs: {belief_dir}", flush=True)
