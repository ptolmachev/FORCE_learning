'''
trying to reproduce Lorenz attractor dynamics without any input
Doesn't work very well, perhaps there is some sort of problem with the initial conditions
'''

import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from src.CT_RNN import CT_RNN
from src.utils.utils import get_project_root
import os

def generate_Lorenz_trajectory(T, dt, sigma=10, beta=8/3, rho=28):
    N_steps = int(np.ceil(T/dt))
    trajectory = np.empty((3, N_steps))
    x_init = np.array([3.09981953, -0.40028133, 26.56966346])
    x = x_init
    for i in range(N_steps):
        rhs = (1.0/100)*np.array([sigma * (x[1] - x[0]),
                       x[0] * (rho - x[2]) - x[1],
                       x[0] * x[1] - beta * x[2]])
        x = x + dt * rhs
        trajectory[:, i] = x
    return trajectory

def scale(signal):
    mx = np.max(signal)
    mn = np.min(signal)
    return 2 * (signal - mn) / (mx - mn) - 1

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
    dt = 0.1  # ms
    num_inputs = 1
    T_train = 10000  # ms

    rnn = CT_RNN(N, num_inps=num_inputs, dt=dt, tau=tau, sr=1.1)

    sim_steps = int(np.ceil(T_train/dt))
    simtime_array = np.arange(sim_steps)*dt

    input_array = np.zeros((num_inputs, sim_steps))

    # periodic output

    target = generate_Lorenz_trajectory(T=T_train, dt=dt)
    fig = plt.figure(figsize=(15, 6))
    plt.plot(simtime_array, target[0, :], color='r', label = r'$x_0$')
    plt.plot(simtime_array, target[1, :], color='g', label = r'$x_1$')
    plt.plot(simtime_array, target[2, :], color='b', label = r'$x_2$')
    plt.legend()
    plt.show()
    target = target[0,:]/10 # so that it roughly fits into -1 to 1 range

    zs, errs, dw_norms = rnn.train(T_train, input_array, target, noise=False)
    print(f"error for the last 100 timesteps: {np.mean(errs[-100:])}")
    rnn.plot_history(list_of_neurons=[0,1,2,3,4,5])

    fig = plot_performance(time=simtime_array, input=input_array[0, :], target=target, z=zs, title="Training")
    img_file = os.path.join(get_project_root(), "imgs", "lorenz_attractor_training")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()

    fig = plot_stats(simtime_array, errs, dw_norms)
    img_file = os.path.join(get_project_root(), "imgs", "lorenz_attractor_training_stats")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()

    # TESTING
    #assumes that T_test < T_train
    T_test = 1000 # ms
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
    img_file = os.path.join(get_project_root(), "imgs", "lorenz_attractor_testing")
    plt.savefig(img_file + ".pdf", ssbbox_inches="tight")
    plt.savefig(img_file + ".png", ssbbox_inches="tight")
    plt.show()
    plt.close()


