'''
This script uses FORCE learning procedure to learn a mapping of a periodic input to the sum of sine waves
the periods of input and the outputs are aligned
'''

import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from src.CT_RNN import CT_RNN
from src.utils.utils import get_project_root
import os

def plot_performance(time, input, target, z, title):
    fig = plt.figure(figsize=(15,6))
    plt.plot(time, target, color ='r', label = 'target')
    plt.plot(time, z, color='b', label = 'output')
    plt.plot(time, input, color='g', label='input', linestyle='--', alpha=0.5)
    plt.xlabel("time, ms", size=24)
    plt.suptitle(title, size=24)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_stats(time, errs, dw_norms):
    fig, ax = plt.subplots(2,1, figsize=(15,10))
    ax[0].plot(time, errs, c='r', label = 'error')
    ax[0].set_xlabel("time, ms", size=24)
    ax[0].set_ylabel("Error", size=24)
    ax[0].legend(fontsize=24)
    ax[0].grid(True)

    ax[1].plot(time, dw_norms, c='b',  label =r'$\|dw\|^2$')
    ax[1].set_xlabel("time, ms", size=24)
    ax[1].set_ylabel(r"$\|dw\|^2$", size=24)
    ax[1].legend(fontsize=24)
    ax[1].grid(True)
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

if __name__ == '__main__':
    N = 1000
    tau = 10  # ms
    dt = 1  # ms
    num_inputs = 1
    T_train = 4*1440  # ms

    rnn = CT_RNN(N, num_inps=num_inputs, dt=dt, tau=tau)

    sim_steps = int(np.ceil(T_train/dt))
    simtime_array = np.arange(sim_steps)*dt

    input_array = np.ones((num_inputs, sim_steps))
    # periodic input
    inp_period = 400 #ms
    input_array[0, :] = np.tanh(5*np.sin(2 * np.pi * (simtime_array / inp_period)))

    # periodic output
    phi_1, phi_2, phi_3, phi_4 = np.pi*np.random.rand(4)
    amp = 1.3
    out_period = 200 #ms
    target = (amp / 1.0) * np.sin(1.0 * np.pi * (1.0/(out_period)) * simtime_array + phi_1) + \
         (amp / 2.0) * np.sin(2.0 * np.pi * (1.0/(out_period)) * simtime_array + phi_2) + \
         (amp / 6.0) * np.sin(3.0 * np.pi * (1.0/(out_period)) * simtime_array + phi_3) + \
         (amp / 3.0) * np.sin(4.0 * np.pi * (1.0/(out_period)) * simtime_array + phi_4)
    target = target / 1.5

    zs, errs, dw_norms = rnn.train(T_train, input_array, target, noise=False)
    print(f"error for the last 100 timesteps: {np.mean(errs[-100:])}")
    rnn.plot_history(list_of_neurons=[0,1,2,3,4,5])

    fig = plot_performance(time=simtime_array, input=input_array[0, :], target=target, z=zs, title="Training")
    img_file = os.path.join(get_project_root(), "imgs", "sine_wave_training")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()
    fig = plot_stats(simtime_array, errs, dw_norms)
    img_file = os.path.join(get_project_root(), "imgs", "sine_wave_training_stats")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()

    # TESTING
    #assumes that T_test < T_train
    T_test = 1440*2 # ms
    sim_steps = int(np.ceil(T_test/dt))
    simtime_array = np.arange(sim_steps)*dt
    input_array = input_array[:, :sim_steps]
    target = target[:sim_steps]
    rnn.reset_v()
    rnn.clear_history()
    rnn.run(T=T_test, input_array=input_array)
    vs = rnn.get_history()
    # get the output as a time sequence
    zs = np.sum((np.hstack([rnn.w_out.reshape(-1, 1)]*vs.shape[-1]) * rnn.activation(vs)), axis = 0)
    fig = plot_performance(time=simtime_array, input=input_array[0,:], target=target, z=zs, title="Test")
    img_file = os.path.join(get_project_root(), "imgs", "sine_wave_testing")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()


