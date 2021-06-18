import numpy as np

def sig(inp: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-inp))

def tanh(inp: np.ndarray) -> np.ndarray:
    return np.tanh(inp)

def softmax(inp: np.ndarray) -> np.ndarray:
    return np.exp(inp) / np.sum(np.exp(inp))

def loss_func(pred: np.ndarray, target:np.ndarray) -> np.ndarray:
     return -np.sum(target * np.log(pred))

class LSTM:
    def __init__(self, str_size, hidden_size):
        # Initialize parameters
        self.str_size = str_size
        self.hidden_size = hidden_size
        
        self.param = {}
        self.param['Wf'] = np.random.randn(hidden_size, (str_size + hidden_size)) / 100
        self.param['bf'] = np.zeros((hidden_size, 1))
        
        self.param['Wi'] = np.random.randn(hidden_size, (str_size + hidden_size)) / 100
        self.param['bi'] = np.zeros((hidden_size, 1))
        
        self.param['Wc'] = np.random.randn(hidden_size, (str_size + hidden_size)) / 100
        self.param['bc'] = np.zeros((hidden_size, 1))
        
        self.param['Wo'] = np.random.randn(hidden_size, (str_size + hidden_size)) / 100
        self.param['bo'] = np.zeros((hidden_size, 1))
        
        self.param['Wy'] = np.random.randn(str_size, hidden_size) / 100
        self.param['by'] = np.zeros((str_size, 1))
        
    def train(self, inputs, target, h_prev, c_prev, learning_rate=1e-2):
        f_t, i_t, c_, c_t = {}, {}, {}, {}
        o_t, h_t, y_t, p_t ={}, {}, {}, {}

        h_t[-1] = h_prev
        c_t[-1] = c_prev
        loss = 0
        
        grad = {}
        for p in self.param:
            grad[p] = np.zeros_like(self.param[p])
        
        # Forward Pass
        for i in range(len(inputs)):
            h_x = np.concatenate((h_t[i - 1], inputs[i]))            # Concatenated Vector (hidden + input)
            f_t[i] = sig(self.param['Wf'] @ h_x + self.param['bf'])  # Forget gate
            i_t[i] = sig(self.param['Wi'] @ h_x + self.param['bi'])   
            c_[i]  = tanh(self.param['Wc'] @ h_x + self.param['bc'])  
            c_t[i] = c_t[i - 1] * f_t[i] + i_t[i] * c_[i]            # Cell State Update
            o_t[i] = sig(self.param['Wo'] @ h_x + self.param['bo'])
            h_t[i] = o_t[i] * tanh(c_t[i])                           # Hidden vector update
            y_t[i] = self.param['Wy'] @ h_t[i] + self.param['by']    # Output
            p_t[i] = softmax(y_t[i])
            loss += loss_func(p_t[i], target[i])
        h_prev = h_t[len(inputs) - 1]
        c_prev = c_t[len(inputs) - 1]
        
        # Backward Pass
        for t in reversed(range(1, len(inputs))):
            h_x = np.concatenate((h_t[t - 1], inputs[t]))
            d_Y = p_t[t] - target[t]
            d_ht = self.param['Wy'].T @ d_Y
            
            grad['Wy'] += d_Y @ h_t[t].T
            grad['by'] += d_Y
            
            grad['Wo'] += (d_ht * tanh(c_t[t]) * o_t[t] * (1 - o_t[t])) @ h_x.T
            grad['bo'] += (d_ht * tanh(c_t[t]) * o_t[t] * (1 - o_t[t]))
            
            d_ct = d_ht * o_t[t] * (1 - tanh(c_t[t]) ** 2)
            
            grad['Wc'] += (d_ct * i_t[t] * (1 - c_t[t] ** 2)) @ h_x.T
            grad['bc'] += (d_ct * i_t[t] * (1 - c_t[t] ** 2))
            
            grad['Wi'] += (d_ct * c_[t] * i_t[t] * (1 - i_t[t])) @ h_x.T
            grad['bi'] += (d_ct * c_[t] * i_t[t] * (1 - i_t[t]))
            
            grad['Wf'] += (d_ct * c_t[t - 1] * f_t[t] * (1 - f_t[t])) @ h_x.T
            grad['bf'] += (d_ct * c_t[t - 1] * f_t[t] * (1 - f_t[t]))
        
        # Mitigating exploding gradients issue
        for dparam in grad:
            np.clip(grad[dparam], -5, 5, out=grad[dparam])
        
        # Optimizer step
        for dparam in grad:
            self.param[dparam] -= learning_rate * grad[dparam]
            
        return loss, h_prev, c_prev
            
    def sample(self, seed_idx, h_prev, c_prev, seq_len):
        x = np.zeros((self.str_size, 1))
        x[seed_idx] = 1
        idx_list = []
        h_t = h_prev
        c_t = c_prev
        for _ in range(seq_len):
            h_x = np.concatenate((h_t, x))
            f_t = sig(self.param['Wf'] @ h_x + self.param['bf'])
            i_t = sig(self.param['Wi'] @ h_x + self.param['bi'])
            c_  = tanh(self.param['Wc'] @ h_x + self.param['bc'])
            c_t = c_t * f_t + i_t * c_
            o_t = sig(self.param['Wo'] @ h_x + self.param['bo'])
            h_t = o_t * tanh(c_t)
            y_t = self.param['Wy'] @ h_t + self.param['by']
            p_t = softmax(y_t)
            ix = np.random.choice(range(self.str_size), p=p_t.ravel())
            x = np.zeros((self.str_size, 1))
            x[ix] = 1
            idx_list.append(ix)
        return idx_list

class LSTM_Peephole(LSTM):
    def __init__(self, str_size, hidden_size):
         # Initialize parameters
        self.str_size = str_size
        self.hidden_size = hidden_size
        
        self.param = {}
        self.param['Wf'] = np.random.randn(hidden_size, (str_size + (hidden_size * 2))) / 100
        self.param['bf'] = np.zeros((hidden_size, 1))
        
        self.param['Wi'] = np.random.randn(hidden_size, (str_size + (hidden_size * 2))) / 100
        self.param['bi'] = np.zeros((hidden_size, 1))
        
        self.param['Wc'] = np.random.randn(hidden_size, (str_size + hidden_size)) / 100
        self.param['bc'] = np.zeros((hidden_size, 1))
        
        self.param['Wo'] = np.random.randn(hidden_size, (str_size + (hidden_size * 2))) / 100
        self.param['bo'] = np.zeros((hidden_size, 1))
        
        self.param['Wy'] = np.random.randn(str_size, hidden_size) / 100
        self.param['by'] = np.zeros((str_size, 1))
        
    def train(self, inputs, target, h_prev, c_prev, learning_rate=1e-2):
        f_t, i_t, c_, c_t = {}, {}, {}, {}
        o_t, h_t, y_t, p_t ={}, {}, {}, {}

        h_t[-1] = h_prev
        c_t[-1] = c_prev
        loss = 0
        
        grad = {}
        for p in self.param:
            grad[p] = np.zeros_like(self.param[p])
        
        # Forward Pass
        for i in range(len(inputs)):
            h_x = np.concatenate((h_t[i - 1], inputs[i]))            # Concatenated Vector (hidden + input)
            h_c_x_ = np.concatenate((c_t[i - 1], h_t[i - 1], inputs[i]))

            f_t[i] = sig(self.param['Wf'] @ h_c_x_ + self.param['bf'])  # Forget gate
            i_t[i] = sig(self.param['Wi'] @ h_c_x_ + self.param['bi'])   
            c_[i]  = tanh(self.param['Wc'] @ h_x + self.param['bc'])  
            c_t[i] = c_t[i - 1] * f_t[i] + i_t[i] * c_[i]            # Cell State Update
            o_t[i] = sig(self.param['Wo'] @ np.concatenate((c_t[i], h_t[i - 1], inputs[i])) + self.param['bo'])
            h_t[i] = o_t[i] * tanh(c_t[i])                           # Hidden vector update
            y_t[i] = self.param['Wy'] @ h_t[i] + self.param['by']    # Output
            p_t[i] = softmax(y_t[i])
            loss += loss_func(p_t[i], target[i])
        h_prev = h_t[len(inputs) - 1]
        c_prev = c_t[len(inputs) - 1]
        
        # Backward Pass
        for t in reversed(range(1, len(inputs))):
            h_x = np.concatenate((h_t[t - 1], inputs[t]))
            h_c_x_ = np.concatenate((c_t[t - 1], h_t[t - 1], inputs[t]))
            d_Y = p_t[t] - target[t]
            d_ht = self.param['Wy'].T @ d_Y
            
            grad['Wy'] += d_Y @ h_t[t].T
            grad['by'] += d_Y
            
            grad['Wo'] += (d_ht * tanh(c_t[t]) * o_t[t] * (1 - o_t[t])) @ np.concatenate((c_t[t], h_t[t - 1], inputs[t])).T
            grad['bo'] += (d_ht * tanh(c_t[t]) * o_t[t] * (1 - o_t[t]))
            
            d_ct = d_ht * o_t[t] * (1 - tanh(c_t[t]) ** 2)
            
            grad['Wc'] += (d_ct * i_t[t] * (1 - c_t[t] ** 2)) @ h_x.T
            grad['bc'] += (d_ct * i_t[t] * (1 - c_t[t] ** 2))
            
            grad['Wi'] += (d_ct * c_[t] * i_t[t] * (1 - i_t[t])) @ h_c_x_.T
            grad['bi'] += (d_ct * c_[t] * i_t[t] * (1 - i_t[t]))
            
            grad['Wf'] += (d_ct * c_t[t - 1] * f_t[t] * (1 - f_t[t])) @ h_c_x_.T
            grad['bf'] += (d_ct * c_t[t - 1] * f_t[t] * (1 - f_t[t]))
        
        # Mitigating exploding gradients issue
        for dparam in grad:
            np.clip(grad[dparam], -5, 5, out=grad[dparam])
        
        # Optimizer step
        for dparam in grad:
            self.param[dparam] -= learning_rate * grad[dparam]
            
        return loss, h_prev, c_prev
            
    def sample(self, seed_idx, h_prev, c_prev, seq_len):
        x = np.zeros((self.str_size, 1))
        x[seed_idx] = 1
        idx_list = []
        h_t = h_prev
        c_t = c_prev
        for _ in range(seq_len):
            h_x = np.concatenate((h_t, x))
            h_c_x_ = np.concatenate((c_t, h_t, x))
            f_t = sig(self.param['Wf'] @ h_c_x_ + self.param['bf'])
            i_t = sig(self.param['Wi'] @ h_c_x_ + self.param['bi'])
            c_  = tanh(self.param['Wc'] @ h_x + self.param['bc'])
            c_t = c_t * f_t + i_t * c_
            o_t = sig(self.param['Wo'] @ np.concatenate((c_t, h_t, x)) + self.param['bo'])
            h_t = o_t * tanh(c_t)
            y_t = self.param['Wy'] @ h_t + self.param['by']
            p_t = softmax(y_t)
            ix = np.random.choice(range(self.str_size), p=p_t.ravel())
            x = np.zeros((self.str_size, 1))
            x[ix] = 1
            idx_list.append(ix)
        return idx_list

class GRU:
    def __init__(self, str_size, hidden_size):
        self.str_size = str_size
        self.hidden_size = hidden_size

        self.param = {}
        self.param['Wz'] = np.random.randn(hidden_size, str_size) / 100
        self.param['Uz'] = np.random.randn(hidden_size, hidden_size) / 100
        self.param['bz'] = np.zeros((hidden_size, 1))

        self.param['Wr'] = np.random.randn(hidden_size, str_size) / 100
        self.param['Ur'] = np.random.randn(hidden_size, hidden_size)/ 100
        self.param['br'] = np.zeros((hidden_size, 1))

        self.param['Wh'] = np.random.randn(hidden_size, str_size) / 100
        self.param['Uh'] = np.random.randn(hidden_size, hidden_size)/ 100
        self.param['bh'] = np.zeros((hidden_size, 1))

        self.param['Wy'] = np.random.randn(str_size, hidden_size) / 100
        self.param['by'] = np.zeros((str_size, 1))

    def train(self, inputs, target, h_prev, learning_rate=0.0001):
        z_t, r_t, h_= {}, {}, {}
        h_t, y_t, p_t = {}, {}, {}
        h_t[-1] = h_prev
        loss = 0

        grad = {}
        for p in self.param:
            grad[p] = np.zeros_like(self.param[p])

        for i in range(len(inputs)):
            z_t[i] = sig(self.param['Wz'] @ inputs[i] + self.param['Uz'] @ h_t[i - 1] + self.param['bz'])
            r_t[i] = sig(self.param['Wr'] @ inputs[i] + self.param['Ur'] @ h_t[i - 1] + self.param['br'])
            h_[i]  = tanh(self.param['Wh'] @ inputs[i] + self.param['Uh'] @ (r_t[i] * h_t[i - 1]) + self.param['bh'])
            h_t[i] = (1 - z_t[i]) * h_t[i - 1] + z_t[i] * h_[i]
            y_t[i] = self.param['Wy'] @ h_t[i] + self.param['by']    # Output
            p_t[i] = softmax(y_t[i])
            loss += loss_func(p_t[i], target[i])
        h_prev = h_t[len(inputs) - 1]

        for t in reversed(range(1, len(inputs))):
            d_Y = p_t[t] - target[t]
            d_ht = self.param['Wy'].T @ d_Y
            
            grad['Wy'] += d_Y @ h_t[t].T
            grad['by'] += d_Y

            grad['Wh'] += (d_ht * z_t[t] * (1 - h_[t] ** 2)) @ inputs[t].T
            grad['Uh'] += (d_ht * z_t[t] * (1 - h_[t] ** 2)) @ (r_t[t] * h_t[t - 1]).T
            grad['bh'] += (d_ht * z_t[t] * (1 - h_[t] ** 2))

            grad['Wr'] += (d_ht * z_t[t] * (1 - h_[t] ** 2) * (self.param['Uh'] @ h_t[t - 1]) * r_t[t] * (1 - r_t[t])) @ inputs[t].T
            grad['Ur'] += (d_ht * z_t[t] * (1 - h_[t] ** 2) * (self.param['Uh'] @ h_t[t - 1]) * r_t[t] * (1 - r_t[t])) @ h_t[t - 1].T
            grad['br'] += (d_ht * z_t[t] * (1 - h_[t] ** 2) * (self.param['Uh'] @ h_t[t - 1]) * r_t[t] * (1 - r_t[t]))

            grad['Wz'] += (d_ht * (h_[t] - h_t[t - 1]) * z_t[t] * (1 - z_t[t])) @ inputs[t].T
            grad['Uz'] += (d_ht * (h_[t] - h_t[t - 1]) * z_t[t] * (1 - z_t[t])) @ h_t[t - 1].T
            grad['bz'] += (d_ht * (h_[t] - h_t[t - 1]) * z_t[t] * (1 - z_t[t]))
        
        # Mitigating exploding gradients issue
        for dparam in grad:
            np.clip(grad[dparam], -5, 5, out=grad[dparam])
        
        # Optimizer step
        for dparam in grad:
            self.param[dparam] -= learning_rate * grad[dparam]
        
        return loss, h_prev


    def sample(self, seed_idx, h_prev, seq_len):
        x = np.zeros((self.str_size, 1))
        x[seed_idx] = 1
        idx_list = []
        h_t = h_prev
        for _ in range(seq_len):
            z_t = sig(self.param['Wz'] @ x + self.param['Uz'] @ h_t + self.param['bz'])
            r_t = sig(self.param['Wr'] @ x + self.param['Ur'] @ h_t + self.param['br'])
            h_ = tanh(self.param['Wh'] @ x + self.param['Uh'] @ (r_t * h_t) + self.param['bh'])
            h_t = (1 - z_t) * h_t + z_t * h_
            y_t = self.param['Wy'] @ h_t + self.param['by']
            p_t = softmax(y_t)
            ix = np.random.choice(range(self.str_size), p=p_t.ravel())
            x = np.zeros((self.str_size, 1))
            x[ix] = 1
            idx_list.append(ix)
        return idx_list
