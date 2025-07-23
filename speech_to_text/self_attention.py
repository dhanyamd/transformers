import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F 

def calculate_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor
): 
    #perform matmul 
    attention_scores = torch.matmul(query, keys.transpose(-2,-1)) 
    attention_scores = attention_scores / math.sqrt(keys.shape[-1]) 
    attention = torch.matmul(attention_scores, values) 
    return attention, attention_scores

class FeedForward(nn.Module): 
    def __init__(self, embed_size: int):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size) 
    def forward(self, x): 
        x = self.layer1(x) 
        x = F.gelu(x)
        x = self.layer2(x) 
        return x 
    
class AttentionLayer(nn.Module): 
    def __init__(self,embed_size: int):
        super().__init__()
        self.embed_size = embed_size 
        self.query_dense = nn.Linear(embed_size, embed_size) 
        self.key_dense = nn.Linear(embed_size, embed_size) 
        self.value_dense = nn.Linear(embed_size, embed_size)
    def forward(self, embeddings: torch.Tensor): 
        query = self.query_dense(embeddings) 
        key = self.key_dense(embeddings) 
        value = self.value_dense(embeddings) 
        attention, attention_scores = calculate_attention(query, key, value)
        return attention, attention_scores
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, E = x.shape  # Batch, Time, Embedding

        # Project and split into heads
        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).reshape(B * self.num_heads, T, self.head_dim)

        Q = split_heads(self.query(x))
        K = split_heads(self.key(x))
        V = split_heads(self.value(x))

        # Calculate attention (your function)
        attn, _ = calculate_attention(Q, K, V)

        # Merge heads
        attn = attn.view(B, self.num_heads, T, self.head_dim).transpose(1, 2).reshape(B, T, E)

        return self.out(attn)
    
class TransformerBlock(nn.Module): 
    def __init__(self, embed_size: int):
        super().__init__()
        self.attention_layer = AttentionLayer(embed_size) 
        self.feed_forward = FeedForward(embed_size) 
        self.layer_norm1 = nn.LayerNorm(embed_size) 
    def forward(self, x: torch.Tensor): 
        context, attention_scores = self.attention_layer(x)
        context = self.layer_norm1(context) 
        context = self.feed_forward(context) 
        context = F.gelu(context) 
        output = context + x 
        return output, attention_scores 
         
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int):
        super().__init__() 
        position = torch.arange(max_seq_length).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(1000.0) / embed_size))
        pe = torch.zeros(max_seq_length, embed_size)  # <-- FIXED HERE
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('position_embedding', pe)
    def forward(self, x: torch.Tensor): 
        return x + self.position_embedding[:x.size(1), :] 


class Transformer(nn.Module):
    def __init__(self, embed_size: int, num_layers: int, max_seq_length: int):
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(
            embed_size, max_seq_length
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)  # Unpack and pass only the tensor output
        return x


if __name__ == "__main__":
    transformer = Transformer(embed_size=128, num_layers=3, max_seq_length=15)
    x = torch.randn(2, 10, 128)
    print(transformer(x).shape)
