import torch
import torch.nn as nn
import math

class SimpleTextRnn(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleTextRnn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class PositionalEncoding(nn.Module):
    # Implement the positional encoding as described in "Attention is All You Need"
    # See positional_encoding_visualisation.ipynb for explanation
    def __init__(self, embed_dimension, max_len=1000):
        super(PositionalEncoding, self).__init__()
        position_encoding_matrix = torch.zeros(max_len, embed_dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, -torch.arange(0, embed_dimension, 2).float() / embed_dimension)
        position_encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        position_encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', position_encoding_matrix.unsqueeze(0))

    def forward(self, x):
        # Take the pe matrix, slicing it to match the given sequence length
        # i.e. We if we've got a max 1000 length context our buffered
        # matrix will be 1000 rows of embed dimension columns
        # But we want to match this to the size of x
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        if embedding_dim % num_heads != 0:
            raise f"embedding_dim {embedding_dim} is not divisible by {num_heads}"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # The dimensions of an individual head, with each head getting an equal part of the embedding dimension
        self.d_k = embedding_dim // num_heads
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

        # The output layer
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dimension):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_dimension)
        self.fc2 = nn.Linear(ff_dimension, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Modified version of https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/
# to make it a decoder-only model
class SimpleTransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dimension, max_length, dropout):
        super(SimpleTransformerDecoderOnly, self).__init__()
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ff_dimension, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    # Make a triangular mask for the decoder self-attention
    # This means that each token can only attend to the previous tokens
    def generate_causal_mask(self, size, device):
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask == 0

    def forward(self, tgt):
        tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        last_position = dec_output[:, -1, :]
        output = self.fc(last_position)
        return output