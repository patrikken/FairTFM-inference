"""
Core model components for FairTFM.
Lightweight inference-only implementation.

The code reuses FeatureEncoder, TargetEncoder, and TransformerEncoderStack from tfmplayground (https://github.com/automl/TFM-Playground) with modifications for fairness and memory efficiency. The main addition is the SensitiveAttributeEncoder which encodes sensitive attributes with learnable padding for test samples.
The forward pass concatenates feature, target, and sensitive attribute embeddings and processes them through the transformer. The decoder then produces logits for the test samples. Memory chunking is applied to attention and MLP computations to reduce memory usage during inference.
"""
import math
import warnings
from typing import Tuple, Callable

import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm


class FeatureEncoder(nn.Module):
    """Encodes tabular features into embeddings."""
    
    def __init__(self, embedding_size: int):
        """Creates the linear layer that will embed features."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """
        Normalizes features based on training data statistics and embeds them.
        
        Args:
            x: Features of shape (batch_size, num_rows, num_features)
            single_eval_pos: Number of training datapoints
            
        Returns:
            Embeddings of shape (batch_size, num_rows, num_features, embedding_size)
        """
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :single_eval_pos], dim=1, keepdims=True)
        std = torch.std(x[:, :single_eval_pos], dim=1, keepdims=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    """Encodes target labels into embeddings."""
    
    def __init__(self, embedding_size: int):
        """Creates the linear layer that will embed targets."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pads training targets to full length using mean and embeds them.
        
        Args:
            y_train: Target labels of shape (batch_size, num_train_datapoints, 1)
            num_rows: Total number of rows
            
        Returns:
            Embeddings of shape (batch_size, num_rows, 1, embedding_size)
        """
        mean = torch.mean(y_train, axis=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class SensitiveAttributeEncoder(nn.Module):
    """Encodes sensitive attributes into embeddings."""
    
    def __init__(self, embedding_size: int, hidden_size=None, learnable_padding=True):
        super().__init__()
        if hidden_size is not None:
            self.linear_layer = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, embedding_size)
            )
        else:
            self.linear_layer = nn.Linear(1, embedding_size)
        
        self.embedding_size = embedding_size
        self.learnable_padding = learnable_padding
        
        ## Learnable padding vector for unseen sensitive attributes 
        self.padding_value = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, s: torch.Tensor, eval_pos: int = 0) -> torch.Tensor:
        """
        Encodes sensitive attributes with padding for test samples.
        
        Args:
            s: Sensitive attributes of shape (batch_size, num_samples)
            eval_pos: Position of evaluation split
            
        Returns:
            Embeddings of shape (batch_size, num_samples, 1, embedding_size)
        """
        if self.learnable_padding:
            padding = self.padding_value.expand(s.shape[0], s[:, eval_pos:].shape[1], 1)
        else:
            mean = torch.mean(s[:, :eval_pos], axis=1, keepdim=True)
            padding = mean.repeat(1, s[:, eval_pos:].shape[1], 1)
        
        s = torch.cat([s[:, :eval_pos], padding], dim=1)
        return self.linear_layer(s.unsqueeze(-1))


class Decoder(nn.Module):
    """Decodes embeddings into class predictions."""
    
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """Initializes the linear layers for decoding."""
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies MLP to embeddings to get logits.
        
        Args:
            x: Embeddings of shape (batch_size, num_rows, embedding_size)
            
        Returns:
            Logits of shape (batch_size, num_rows, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))


def memory_chunking(num_mem_chunks: int) -> callable:
    """
    Decorator to split computation into chunks to reduce memory usage.
    
    Args:
        num_mem_chunks: Number of chunks to split into
    """
    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if num_mem_chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn("Memory chunking disabled during gradient computation for correct gradients.")
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / num_mem_chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(x_split)
            return x
        return wrapper
    return decorator


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with attention between features and datapoints."""

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """
        Forward pass with feature and datapoint attention.
        
        Args:
            src: Embeddings of shape (batch_size, num_rows, num_features, embedding_size)
            single_eval_position: Position of training/test split
            num_mem_chunks: Number of memory chunks for computation
            
        Returns:
            Transformed embeddings of same shape as input
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # Attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            return self.self_attention_between_features(x, x, x)[0] + x
        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # Attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            x_left = self.self_attention_between_datapoints(
                x[:, :single_eval_position], x[:, :single_eval_position], x[:, :single_eval_position]
            )[0]
            x_right = self.self_attention_between_datapoints(
                x[:, single_eval_position:], x[:, :single_eval_position], x[:, :single_eval_position]
            )[0]
            return torch.cat([x_left, x_right], dim=1) + x
        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        
        # MLP
        src = src.reshape(-1, embedding_size)
        @memory_chunking(num_mem_chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x
        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm3(src)
        return src


class TransformerEncoderStack(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            )

    def forward(self, x: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """
        Apply all transformer blocks sequentially.
        
        Args:
            x: Embeddings of shape (batch_size, num_rows, num_features, embedding_size)
            single_eval_position: Position of training/test split
            num_mem_chunks: Number of memory chunks
            
        Returns:
            Transformed embeddings
        """
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, num_mem_chunks=num_mem_chunks)
        return x


class FairTFM(nn.Module):
    """
    Fair Transformer Foundation Model for tabular data with sensitive attribute awareness.
    
    Lightweight inference-only model that makes fair predictions by encoding sensitive
    attributes alongside features and targets. Supports batch prediction over tabular data.
    
    Args:
        embedding_size (int): Dimensionality of embeddings
        num_attention_heads (int): Number of attention heads in transformer
        mlp_hidden_size (int): Hidden dimension of MLP layers in transformer
        num_layers (int): Number of transformer layers
        num_outputs (int): Number of output classes
        sensitive_attr_hidden_size (int, optional): Hidden size of sensitive attribute encoder
    """
    
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        sensitive_attr_hidden_size: int = None,
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        
        # Model components
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.sensitive_attr_encoder = SensitiveAttributeEncoder(
            embedding_size, hidden_size=sensitive_attr_hidden_size
        )
        self.transformer_encoder = TransformerEncoderStack(
            num_layers, embedding_size, num_attention_heads, mlp_hidden_size
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
    
    def forward(
        self,
        src: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        single_eval_pos: int,
        return_embeddings: bool = False,
        num_mem_chunks: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            src (Tuple): (X, y, s) where
                - X: Features of shape (batch_size, num_samples, num_features)
                - y: Target labels of shape (batch_size, num_samples)
                - s: Sensitive attributes of shape (batch_size, num_samples)
            single_eval_pos (int): Position of train/test split
            return_embeddings (bool): Whether to return transformer embeddings
            num_mem_chunks (int): Number of memory chunks for computation
        
        Returns:
            Logits of shape (batch_size, num_test_samples, num_outputs)
            If return_embeddings=True, also returns transformer embeddings
        """
        x_src, y_src, sensitive_attr_src = src
        
        # Ensure proper shapes
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        if len(sensitive_attr_src.shape) < len(x_src.shape):
            sensitive_attr_src = sensitive_attr_src.unsqueeze(-1)
        
        # Encode all components
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        sensitive_attr_src = self.sensitive_attr_encoder(sensitive_attr_src, single_eval_pos)
        
        # Concatenate and pass through transformer
        src = torch.cat([x_src, y_src, sensitive_attr_src], 2)
        src_embeddings = self.transformer_encoder(
            src, single_eval_pos, num_mem_chunks=num_mem_chunks
        )
        
        # Extract embeddings for test samples
        decoder_input_embeddings = src_embeddings[:, single_eval_pos:, -1, :]
        logits = self.decoder(decoder_input_embeddings)
        
        if return_embeddings:
            return logits, src_embeddings
        
        return logits
