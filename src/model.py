import torch
from torch import nn
import config


class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #embedding是默认初始化的
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x):
        # x.shape[batch_szie,seq_len],sql_len=token长度
        embed = self.embedding(x)
        # embed.shape:[batch_size,sql_len,embedding_dim]
        output, _ = self.rnn(embed)
        # output.shape:[batch_size,sql_len,hidden_dim]
        #seq_len 的维度是在计算过程中被聚合到了 last_hidden 的表示中，而不是简单地“消失”了。

        last_hidden = output[:, -1, :]
        #从所有时间步的输出中，单独提取出最后一个时间步（-1）的输出
        # last_hidden.shape:[batch_size,hidden_dim]

        output = self.linear(last_hidden)
        # output.shape[batch_size,vocab_size]
        return output
