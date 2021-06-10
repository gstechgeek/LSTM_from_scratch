import numpy as np
from lstm import LSTM

data = open('input.txt', 'r').read()
chars = list(set(data))
vocab_size = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# convert char sequence to one-hot-encoding
def str2vec(inp_str):
    myenc = []
    for ch in inp_str:
        vec = np.zeros((vocab_size, 1))
        vec[char_to_ix[ch]] = 1
        myenc.append(vec)
    return myenc

def vec2str(ix_list):
    return ''.join([ix_to_char[i] for i in ix_list])

if __name__=='__main__':
    ptr = 0
    seq_len = 25
    hidden_size = 30
    lstm_block = LSTM(str_size=vocab_size, hidden_size=hidden_size)

    # Train and Sample every 100 steps
    for epoch in range(100000):
        if ptr + seq_len + 1 > len(data) or epoch == 0:
            h_prev = np.zeros((hidden_size, 1))
            c_prev = np.zeros((hidden_size, 1))
            ptr = 0

        inputs = str2vec(data[ptr:ptr + seq_len])
        loss, h_prev, c_prev = lstm_block.train(inputs, h_prev, c_prev, learning_rate=5e-3)
        ptr += seq_len
        
        if epoch % 100 == 0:
            idx_list = lstm_block.sample(char_to_ix[data[ptr]], h_prev, c_prev, 100)
            print('----------% Epoch: {}, Loss: {}%------------'.format(epoch, loss))
            print(vec2str(idx_list))
            print()
    