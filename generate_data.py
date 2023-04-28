#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from utils import read_specs, RK4, RK6
from run_da import path_to_specs
from def_dyn import get_dynamics

if __name__ == '__main__':
    ########### MODIFY HERE ###########
    #CaFluorescence
    # dt = 0.025 # time step of the model
    # num_data = 40000 # number of time steps to generate
    # x0 = np.random.rand(5) # initial condition
    # p = (100.0, 450.0, 50.0, 19.0, 2.7, 6.0, 2.0, 50.0, -90.0, -70.0, 85.0, -65.0, -7.8, 0.4, -0.1, -67.0, 2.0, -35.0, -5.0, -20.0, -0.05, -30.0, -5.0, 200.0, 87.5, 0.3, 0.5, 0.1, 1.0, 10.0, ) # parameters, must be in tuple
    # noise_std = np.sqrt(1) # add noise to the data

    #Jason_CaFluorescence
    dt = 0.01 # time step of the model
    num_data = 40000 # number of time steps to generate
    x0 = np.random.rand(5) # initial condition
    p = (10.0, 1.0, 40.0, 17.5, 68.0, 45.0, -90.0, -70.0, 85.0, -35.0, -30.0, -37.4, -5.0, -5.0, 4.3, -30.0, -12.0, -65.0, -7.8, 0.4, -0.1, -68.0, 2.2, 2.2, 1.0, -105.0, 5.0, -105.0, 25.0, 0.35, 0.35, 100.0, 0.1, 0.0015, 0.3, 450.0, 50.0, 2.0, 3.0, 10.0, 2.0, ) # parameters, must be in tuple
    noise_std = np.sqrt(1) # add noise to the data

    #NaKL
    # dt = 0.025 # time step of the model
    # num_data = 40000 # number of time steps to generate
    # x0 = np.random.rand(4) # initial condition
    # p = (1, 120, 20, 0.3, 50, -77, -54.4,\
    #     -40, -60, -55, 15, -15, 30, 0.1,\
    #     0.4, 1, 7, 1, 5,) # parameters, must be in tuple
    # noise_std = np.sqrt(1) # add noise to the data
    #L96
    # dt = 0.025 # time step of the model
    # num_data = 1000 # number of time steps to generate
    # x0 = 20*np.random.rand(5)-10 # initial condition
    # p = (8.17,) # parameters, must be in tuple
    # noise_std = np.sqrt(0.25) # add noise to the data
    ####################################

    specs = read_specs(path_to_specs)
    dyn, _, _ = get_dynamics(specs)
    if not os.path.exists(specs['data_folder']): os.makedirs(specs['data_folder'])

    f = lambda r, t, params: np.array(dyn(r, t, *params))
    if specs.get('stim_file') is not None:
        stim = 7 * np.load(specs['data_folder']+specs['stim_file'])[:, 1] # amplify the stim
    else:
        stim = np.empty(num_data)

    time_arr = np.linspace(0, num_data*dt, num_data)
    sol = np.zeros((num_data, specs['num_dims']))
    sol[0] = x0
    for i, t in enumerate(time_arr[1:], 1):
        sol[i] = RK6(f, sol[i-1], t, dt, params = (p, stim[i-1]))
        print("generated data at time=", t)
    # interp_stim = interp1d(time_arr, stim)
    # solution = solve_ivp(f, time_arr, x0, method='RK45', t_eval=time_arr, args=(p, interp_stim, ), vectorized=True)

    # plt.plot(time_arr, sol[:, 0])
    # plt.show()

    obs_dim = specs['obs_dim'] if specs['obs_dim'] != -1 else np.arange(specs['num_dims'])
    np.save(specs['data_folder']+specs['data_file'],
               np.vstack((time_arr, sol.T[obs_dim]+np.random.normal(0, noise_std, (sol.T[obs_dim].shape)))).T)
    np.save(specs['data_folder']+'all_'+specs['data_file'], np.vstack((time_arr, sol.T)).T)








