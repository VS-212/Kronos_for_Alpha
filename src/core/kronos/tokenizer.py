"""
M-TOKENIZE: Kronos VQ-VAE tokenizer
Contract: OHLCV tensor [B,T,6] → token_ids [B,T], recon [B,T,6]
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.core.kronos.modules import (
    BSQuantizer,
    TransformerBlock,
)


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    KronosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(
        self,
        d_in,
        d_model,
        n_heads,
        ff_dim,
        n_enc_layers,
        n_dec_layers,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        s1_bits,
        s2_bits,
        beta,
        gamma0,
        gamma,
        zeta,
        group_size,
    ):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits  # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.enc_layers - 1)
            ]
        )
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.dec_layers - 1)
            ]
        )
        self.quant_embed = nn.Linear(
            in_features=self.d_model, out_features=self.codebook_dim
        )  # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(
            in_features=self.s1_bits, out_features=self.d_model
        )  # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(
            in_features=self.codebook_dim, out_features=self.d_model
        )  # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size
        )  # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the KronosTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)  # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[
            :, :, : self.s1_bits
        ]  # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0]  # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(
                self.codebook_dim // 2, device=x1.device, dtype=torch.long
            )  # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0  # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0  # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1)  # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(
                self.codebook_dim, device=x.device, dtype=torch.long
            )  # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0  # Extract bits

        x = x.float() * 2 - 1  # Convert boolean to bipolar (-1, 1)
        q_scale = 1.0 / (self.codebook_dim**0.5)  # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z
