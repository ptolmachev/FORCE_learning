'''
A script containing a continuous-time RNN with feedback loop
governed by the equations:
tau dv/dt = -v + W_rec * sigma(v) + W_inp * sigma(u) + W_fb * sigma(z) + b
z = W_out @ sigma(v)
# For now output z is a scalar!
sigma(h) function described in 'state_function.py'
'''
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import eigs
from tqdm.auto import tqdm
from scipy.stats import uniform

def generate_recurrent_weights(N, density, sr):
    A = (1.0/(density * np.sqrt(N))) * np.array(random(N, N, density, data_rvs=uniform(-1, 2).rvs).todense())
    #get eigenvalues
    w, v = eigs(A)
    A = A * (sr/np.max(np.abs(w)))
    return A

class CT_RNN():
    def __init__(self, N, num_inps, num_outs, dt, tau=25, sr=0.9, maxlen=1000000, bias=False, sparcity_param = 0.1,
                 input_scaling=1, fb_scaling=1):
        self.maxlen = maxlen
        self.tau = tau
        self.dt = dt
        self.N = N
        self.num_inps = num_inps
        self.num_outs = num_outs
        self.input_scaling = input_scaling
        self.fb_scaling = fb_scaling
        self.sr = sr
        self.sparcity_param = sparcity_param

        W_rec = generate_recurrent_weights(self.N, density=self.sparcity_param, sr=self.sr)
        W_inp = self.input_scaling * (2 * np.random.rand(N, num_inps) - 1)
        W_fb = self.fb_scaling * (2 * np.random.rand(N, num_outs) - 1)
        W_out = 1 / (np.sqrt(self.N)) * np.random.rand(num_outs, N)

        self.W_rec = W_rec
        self.W_inp = W_inp
        self.W_fb = W_fb
        self.W_out = W_out

        if bias == False:
            self.b = np.zeros(self.N)
        else:
            self.b = 0.1 * np.random.randn(self.N)

        self.v_init = 0.01 * np.random.randn(self.N)
        self.v = self.v_init

        self.t = 0
        self.activation = lambda x: np.tanh(x)
        self.v_history = deque(maxlen=maxlen)

    def rhs(self, v, inp_vect, noise_amp):
        z = self.W_out @ self.activation(v)
        return (1.0/self.tau) * (-v +
                                 (self.W_rec @ self.activation(v)
                                  + self.W_inp @ self.activation(inp_vect)
                                  + self.W_fb @ (self.activation(z) + noise_amp * np.random.randn(self.num_outs))
                                  + self.b))

    def step(self, inp_vect, noise_amp):
        # k1 = self.dt * self.rhs(self.v, inp_vect, noise_amp)
        # k2 = self.dt * self.rhs(self.v + k1 / 2, inp_vect, noise_amp)
        # k3 = self.dt * self.rhs(self.v + k2 / 2, inp_vect, noise_amp)
        # k4 = self.dt * self.rhs(self.v + k3, inp_vect, noise_amp)
        # self.v = self.v + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.v = self.v + self.dt * self.rhs(self.v, inp_vect, noise_amp)
        return None

    def update_history(self):
        self.v_history.append(deepcopy(self.v))
        self.t += self.dt
        return None

    def clear_history(self):
        self.v_history = deque(maxlen=self.maxlen)
        return None

    def reset_v(self):
        self.v = self.v_init
        return None

    def run(self, T, input_array, noise_amp=0):
        #input array must be the length of int(np.ceil(T/self.dt))!
        N_steps = int(np.ceil(T/self.dt))
        for i in (range(N_steps)):
            inp_vect = input_array[:, i]
            self.step(inp_vect, noise_amp=0)
            self.update_history()
        return None

    def train(self, T, input_array, target_array, noise_amp):
        # both input_array and target_array must be the length of int(np.ceil(T/self.dt))!
        N_steps = int(np.ceil(T / self.dt))
        # initialize estimate of inverse hessian matrix
        self.P = np.eye(self.N)
        # initialize buffers for useful statistics
        self.error_buffer = []
        self.dw_norm_buffer = []
        self.z_buffer = []

        for i in tqdm(range(N_steps)):
            inp_vect = input_array[:, i]
            target = target_array[i] #scalar for now
            self.step(inp_vect, noise_amp=noise_amp)

            z, e, dW = self.get_weight_update(target)
            self.W_out = deepcopy(self.W_out + dW)

            self.error_buffer.append(e**2)
            self.dw_norm_buffer.append(np.linalg.norm(dW))
            self.z_buffer.append(z)
            self.update_history()
        return self.z_buffer, self.error_buffer, self.dw_norm_buffer

    def get_weight_update(self, target):
        # update the error for the linear readout: y_t - r_t^T w_{out}_{t-1}
        # where y_t is the target at time t,
        # r_t is the vector of neural firing rates at time t,
        # and the \w_{out}_{t-1} - readout weights
        # WARNING
        # WORKS ONLY WITH num_outs = 1 for now!
        r = self.activation(self.v)
        z = self.W_out @ r # output signal
        e = (target - z)

        # # update an estimate of (X^T X)^{-1} matrix:
        # P_{t} = P_{t-1} - ((P_{t-1} r_t) (r_t^T P_{t-1})) / (1 + r_t^T P_{t-1} r_t)
        Pr = (self.P @ r.reshape(-1, 1)).flatten()
        rPr = np.sum(r * Pr)
        c = 1.0 / (1 + rPr)
        self.P = self.P - c * (Pr.reshape(-1, 1) @ Pr.reshape(1, -1))

        # update the output weights
        dw = e * Pr * c
        return z, e, dw

    def get_history(self):
        v_array = np.array(self.v_history)
        return v_array.T

    def plot_history(self, list_of_neurons=None):
        transients = 100
        if list_of_neurons is None:
            v_array = self.get_history()[:,transients:]
        else:
            v_array = self.get_history()[list_of_neurons,transients:]
        num_neurons = v_array.shape[0]

        fig, ax = plt.subplots(num_neurons, 1, figsize=(15, num_neurons*1))
        t_array = np.arange(v_array.shape[-1]) * self.dt
        for i in range(num_neurons):
            ax[i].plot(t_array, v_array[i, :], linewidth=2, color='k')
            ax[i].set_yticks([])
            if (i == num_neurons//2):
                ax[i].set_ylabel(f'v', fontsize=24, rotation=0)
        ax[-1].set_xlabel('t', fontsize=24)
        plt.subplots_adjust(hspace=0)
        plt.suptitle(f"Trajectory of a neural network, N={self.N}, tau={self.tau}, dt={self.dt}", fontsize=24)
        return fig, ax

if __name__ == '__main__':
    N = 100
    tau = 25 #ms
    dt = 1 #ms
    num_inputs = 2
    num_outs = 1
    T = 4000 #ms

    rnn = CT_RNN(N, num_inps=num_inputs, num_outs=num_outs, dt=dt, tau=tau, sr=1.2, input_scaling=3)
    #periodic input
    period = 200 #ms
    phi = 2 * np.pi * np.random.rand()
    input_array = np.zeros((num_inputs, int(np.ceil(T/dt))))
    input_array[0, :] = np.sin(2 * np.pi * np.arange(int(np.ceil(T / dt))) / (period))
    input_array[1, :] = np.sin(2 * np.pi * np.arange(int(np.ceil(T / dt))) / (2 * period) + phi)

    rnn.run(T, input_array)
    fig, ax = rnn.plot_history(list_of_neurons=np.arange(5))
    plt.show(block=True)
