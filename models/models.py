import sys

import simplems.ops

sys.path.append('./python')
import simplems as ndl
import simplems.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(nn.Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()
        self.moudle = nn.Sequential(
            nn.Conv(a, b, k, s, device=device, dtype=dtype),
            nn.BatchNorm2d(b, device=device, dtype=dtype),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.moudle(x)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
         ###
        self.ConvBN0 = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        )
        self.ConvBN1 = nn.Residual(
            nn.Sequential(
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            )
        )
        self.ConvBN2 = nn.Sequential(
            ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        )
        self.ConvBN3 = nn.Residual(
            nn.Sequential(
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            )
        )
        self.Linear0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
        

    def forward(self, x):
        
        x = self.ConvBN0(x)
        x = self.ConvBN1(x)
        x = self.ConvBN2(x)
        x = self.ConvBN3(x)
        x = self.Linear0(x)
        return x
        


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            seq_model = nn.RNN
        else:
            seq_model = nn.LSTM
        self.seq_model = seq_model(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        
        l, b = x.shape
        embedding = self.embedding(x)
        # l * b, d
        feature, h = self.seq_model(embedding, h)
        d = feature.shape[-1]
        # l, b, d -> l * b, d
        feature = simplems.ops.reshape(feature, (l * b, d))
        # l * b, d
        output = self.out_proj(feature)

        return output, h
        


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)