'''
A script containing a continuous-time RNN with feedback loop
governed by the equations:
tau dv/dt = -v + W_rec * sigma(v) + W_inp * sigma(u) + w_fb * sigma(z) + b
z = w_out @ sigma(v)
# For now output z is a scalar!
sigma(h) function described in 'state_function.py'
'''
from reservoirpy.mat_gen import generate_input_weights, generate_internal_weights
from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
import numpy as np
from tqdm.auto import tqdm

class CT_RNN():
    def __init__(self, N, num_inps, dt, tau, maxlen=1000000, bias=False):
        self.maxlen = maxlen
        self.tau = tau
        self.dt = dt
        self.N = N
        self.num_inps = num_inps

        sparcity_param = 0.4
        W_rec = np.array(generate_internal_weights(N, proba=sparcity_param, sr=1.5).todense())
        W_inp = generate_input_weights(N, num_inps, proba=sparcity_param, input_scaling=3, input_bias=False)
        w_out = np.zeros(N)
        # w_fb = generate_input_weights(N, 1, proba=sparcity_param, input_scaling=3, input_bias=False).flatten()
        w_fb = 2.0 * (np.random.rand(N) - 0.5)

        self.W_rec = W_rec
        self.W_inp = W_inp
        self.w_out = w_out
        self.w_fb = w_fb

        if bias == False:
            self.b = np.zeros(self.N)
        else:
            self.b = 0.1 * np.random.randn(self.N)

        self.v_init = 0.01 * np.random.randn(self.N)
        self.v = self.v_init

        self.t = 0
        self.activation = lambda x: (1+np.tanh(x))/2.0
        # self.activation = lambda x: np.tanh(x)
        self.v_history = deque(maxlen=maxlen)

    def rhs(self, inp_vect, noise):
        z = np.sum(self.w_out * self.activation(self.v)) # output signal
        return (1.0/self.tau) * (-self.v +
                                 (self.W_rec @ self.activation(self.v)
                                  + self.W_inp @ self.activation(inp_vect)
                                  + self.w_fb * (self.activation(z) + noise * 0.1 * np.random.randn(self.N)) #feedback loop
                                  + self.b))

    def step(self, inp_vect, noise=False):
        self.v = self.v + self.dt * self.rhs(inp_vect, noise)
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

    def run(self, T, input_array):
        #input array must be the length of int(np.ceil(T/self.dt))!
        N_steps = int(np.ceil(T/self.dt))
        for i in (range(N_steps)):
            inp_vect = input_array[:, i]
            self.step(inp_vect)
            self.update_history()
        return None

    def train(self, T, input_array, target_array):
        #number of different sequence to sequence mappings the network has to learn
        # num_maps = target_array.shape[0]
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
            target = target_array[i] # scalar for now
            self.step(inp_vect, noise=True)
            z, e, dw = self.update_weights(target)
            self.error_buffer.append(e**2)
            self.dw_norm_buffer.append(np.linalg.norm(dw))
            self.z_buffer.append(z)
            self.update_history()
        return self.z_buffer, self.error_buffer, self.dw_norm_buffer

    def update_weights(self, target):
        r = self.activation(self.v)

        # update an estimate of inverse hessian matrix
        Pr = (self.P @ r.reshape(-1, 1)).flatten()
        rPr = np.sum(r * Pr)
        c = 1.0 / (1 + rPr)
        self.P = self.P - c * (Pr.reshape(-1, 1) @ Pr.reshape(1, -1))

        # update the error for the linear readout
        z = np.sum(self.w_out * self.activation(self.v)) # output signal
        e = (z - target)
        # update the output weights
        dw = -e * Pr * c
        self.w_out = deepcopy(self.w_out + dw)
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
    tau = 10 #ms
    dt = 1 #ms
    num_inputs = 2
    T = 2000 #ms

    rnn = CT_RNN(N,num_inps=num_inputs, dt=dt, tau=tau)

    #periodic input
    period = 200 #ms
    phi = 2 * np.pi * np.random.rand()
    input_array = np.zeros((num_inputs,int(np.ceil(T/dt))))
    input_array[0, :] = np.sin(2 * np.pi * np.arange(int(np.ceil(T / dt))) / (period))
    input_array[1, :] = np.sin(2 * np.pi * np.arange(int(np.ceil(T / dt))) / (2 * period) + phi)

    # constant input
    # input_array = np.zeros((num_inputs,int(np.ceil(T/dt))))
    # input_array = 0.01*np.ones((num_inputs, int(np.ceil(T/dt))))

    rnn.run(T, input_array)
    fig, ax = rnn.plot_history(list_of_neurons=[0,1,2,3,4])
    plt.show(block=True)
