import torch
import torch.nn as nn

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
    def __init__(self, embed_dimension, max_len=1000):
        super(PositionalEncoding, self).__init__()
        position_encoding_matrix = torch.zeros(max_len, embed_dimension)

        # Create a vector of values from 0 to embed_dimension
        position = torch.arange(0, embed_dimension, 1)

        # Convert to pytorch floats
        position = position.float()

        # Turn it into a 1D tensor
        position = position.unsqueeze(1)


class SimpleTransformer(nn.Transformer):
    self.position_encoding = PositionalEncoding(ninp, dropout)