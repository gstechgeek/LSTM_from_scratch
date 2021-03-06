import json
import numpy as np
import matplotlib.pyplot as plt
# from lstm_archs import LSTM_Peephole as LSTM
from lstm_archs import GRU

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
    smooth_loss = -np.log(1.0 / vocab_size) * seq_len
    # lstm_block = LSTM(str_size=vocab_size, hidden_size=hidden_size)
    gru_block = GRU(str_size=vocab_size, hidden_size=hidden_size)

    loss_lst = []
    smooth_loss_lst = []

    plt.ion()
    plt.figure(figsize=(10, 5))
    # Train and Sample every 100 steps
    for epoch in range(100000):
        if ptr + seq_len + 1 > len(data) or epoch == 0:
            h_prev = np.zeros((hidden_size, 1))
            # c_prev = np.zeros((hidden_size, 1))
            ptr = 0

        inputs = str2vec(data[ptr:ptr + seq_len])
        target = str2vec(data[ptr + 1:ptr + seq_len + 1])
        # 
        loss, h_prev = gru_block.train(inputs, target, h_prev, learning_rate=5e-3)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        ptr += seq_len

        smooth_loss_lst.append(smooth_loss)
        loss_lst.append(loss)
        
        if epoch % 100 == 0:
            # idx_list = lstm_block.sample(char_to_ix[data[ptr]], h_prev, c_prev, 200)
            idx_list = gru_block.sample(char_to_ix[data[ptr]], h_prev, 200)
            print('----------% Epoch: {}, Loss: {}%------------'.format(epoch, smooth_loss))
            print(vec2str(idx_list))
            print()

            plt.plot(loss_lst, label='loss')
            plt.plot(smooth_loss_lst, label='smooth_loss')
            plt.title('Loss Graph: Sequence Length: {}'.format(seq_len))
            plt.xlabel('Num Iterations')
            plt.ylabel('Loss')
            plt.legend(loc='upper right') 
            plt.savefig('lg_sl_{}.png'.format(seq_len))

            plt.draw()
            plt.pause(1e-5)
            plt.clf()
    
    # idx_list = lstm_block.sample(char_to_ix[data[ptr]], h_prev, c_prev, 5000)
    idx_list = gru_block.sample(char_to_ix[data[ptr]], h_prev, 5000)
    out = vec2str(idx_list)

    with open('gru_output.txt', 'w') as f:
        f.write(out)
    # init_params = {'vocab_size': vocab_size, 'hidden_size': hidden_size}
    # weight_dict = {k: v.tolist() for k, v in lstm_block.param.items()}

    # save_dict = {'init': init_params, 'weights': weight_dict}

    # with open('init_weights_peephole.json', 'w') as outfile:
    #     json.dump(save_dict, outfile)
    # print("Weights saved.")
        
        
    