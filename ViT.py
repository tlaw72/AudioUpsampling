# Code inspiration taken from the ViT paper github: https://github.com/google-research/vision_transformer/tree/main

import torch
import torch.nn as nn
import os


# override the timm package to relax the input shape constraint.

class Embedding(nn.Module):
    def __init__(self, img_size=(2220, 5820), patch_size=185, in_channels=4, emb_size=256, dropout_rate=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.num_patches = (self.img_height/self.patch_size) * (self.img_width/self.patch_size)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Patch Embedding

        x = self.projection(x)
        print(f'x shape after proj: {x.shape}')
        # x = x.permute(1,0,2)
        # Position Embedding
        print(f'x shape after transpose: {x.shape}')
        pe = nn.init.normal_(torch.empty(1, x.shape[1], x.shape[2]), std=0.02)
        print(f'pe shape: {pe.shape}')
        x = x + pe
        x = nn.Dropout(p=self.dropout_rate)(x)
        return x


class MlpBlock(nn.Module):

  def __init__(self, mlp_dim=2048, out_dim=768, dropout_rate=0.1):
    super().__init__()
    self.mlp_dim = mlp_dim
    self.out_dim = out_dim
    self.dropout_rate = dropout_rate


#   mlp_dim: int
#   dtype: Dtype = jnp.float32
#   out_dim: Optional[int] = None
#   dropout_rate: float = 0.1
#   kernel_init: Callable[[PRNGKey, Shape, Dtype],
#                         Array] = nn.initializers.xavier_uniform()
#   bias_init: Callable[[PRNGKey, Shape, Dtype],
#                       Array] = nn.initializers.normal(stddev=1e-6)

  
  def forward(self, inputs):

    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    print(f'MLP inputs shape: {inputs.shape}')
    # print(f'MLP actual out shape: {actual_out_dim}')
    # print(f'MLP mlp_dim shape: {self.mlp_dim}')
    x = nn.Linear(
        in_features=actual_out_dim,
        out_features=self.mlp_dim,
        )(inputs)
    nn.GELU(x)
    x = nn.Dropout(p=self.dropout_rate)(x)
    output = nn.Linear(
        in_features=self.mlp_dim,
        out_features=actual_out_dim
        )(x)
    output = nn.Dropout(p=self.dropout_rate)(output)
    return output


class Encoder1DBlock(nn.Module):

  def __init__(self, mlp_dim, num_heads, out_dim, dropout_rate, attention_dropout_rate):
    super().__init__()
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.out_dim = out_dim
  
#   num_heads: int
#   dtype: Dtype = jnp.float32
#   dropout_rate: float = 0.1
#   attention_dropout_rate: float = 0.1

  def forward(self, inputs):
    
    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(normalized_shape=(256,inputs.shape[-1]))(inputs)
    x, x_weights = nn.MultiheadAttention(
        embed_dim=2340,
        dropout=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x, x)
    x = nn.Dropout(p=self.dropout_rate)(x)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(normalized_shape=(256,x.shape[-1]))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, out_dim=self.out_dim, dropout_rate=self.dropout_rate)(
            y)

    # Includes skip connection
    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """
  def __init__(self, mlp_dim, out_dim, num_layers, num_heads, dropout_rate, attention_dropout_rate):
    super().__init__()
    self.mlp_dim = mlp_dim
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.attention_dropout_rate = attention_dropout_rate

#   num_layers: int
#   mlp_dim: int
#   num_heads: int
#   dropout_rate: float = 0.1
#   attention_dropout_rate: float = 0.1
#   add_position_embedding: bool = True

  def forward(self, x):

    assert x.ndim == 3  # (batch, len, emb)

    # Input Encoder
    for layer in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          out_dim=self.out_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          num_heads=self.num_heads)(x)
    encoded = nn.LayerNorm(normalized_shape=(256,x.shape[-1]))(x)

    return encoded




#   num_classes: int
#   patches: Any
#   transformer: Any
#   hidden_size: int
#   resnet: Optional[Any] = None
#   representation_size: Optional[int] = None
#   classifier: str = 'token'
#   head_bias_init: float = 0.
#   encoder: Type[nn.Module] = Encoder
#   model_name: Optional[str] = None

class VisionTransformer(nn.Module):

    def __init__(self, in_channels=4, patch_size=185, emb_size=256, img_size=(2220, 5820), num_heads=8, num_layers=6, batch_size=2):
        super().__init__()

        self.patch_embedding = Embedding(img_size, patch_size, in_channels, emb_size, 0.1)
        self.encoder = Encoder(mlp_dim=2048, out_dim=2340, num_layers=num_layers, num_heads=num_heads, dropout_rate=0.1, attention_dropout_rate=0.1)
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=4, padding=1)
        self.batch_size = batch_size
        self.emb_size = emb_size
        # self.num_layers = num_layers

        # self.mlp_head = nn.Sequential(
        #     nn.Linear(emb_size, num_classes),
        #     nn.Unflatten(1, (num_classes, img_size // patch_size, img_size // (4*patch_size))),
        #     nn.Upsample(size=(600,150))

        # )



    def forward(self, x: torch.Tensor):
        print(f'starting the embedding. input has shape {x.shape}')
        x = self.patch_embedding(x)
        print(f"embedding worked I guess. new shape: {x.shape}")

        x = self.encoder(x)
        print(f"after encoding. x.shape: {x.shape}")
        x = x.reshape(self.batch_size, self.emb_size, 10, 234)
        x = self.upsample(x)
        print(f"done! x.shape: {x.shape}")
        # TODO: Upsample time
        # x = self.mlp_head(x)
        return x
