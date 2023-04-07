#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''

import numpy as np
from numba import njit
import os
import scipy
from utils import RK4

try: import cyipopt
except: print('ipopt not installed')

try: from snopt import snoptb
except: print('snopt not installed')

class Action:
    '''Class to perform the action minimization routine'''

    def __init__(self, params, seed):
        '''initialize the action object

        Args:
            params  : parameters including the dynamics and
            seed    : np seed sequence
        '''
        self.rng = np.random.default_rng(seed)
        self.name = params.get('name')
        self.data_folder = params['data_folder']

        N_data = params['ndata'] if params['ndata'] > 0 else None
        path_to_stim = self.data_folder+params.get('stim_file') if params.get('stim_file') is not None else None
        self.set_data_fromfile(self.data_folder+params['data_file'],
                               path_to_stim,
                               nstart = params.get('nstart', 0),
                               N = N_data)
        self.action_init(params)


    def action_init(self, params):
        '''initialize params from dict
        Args:
            params  : dictionary of params for action
        '''
        self.f              = params.get('f')
        self.fjacx          = params.get('fjacx')
        self.fjacp          = params.get('fjacp')
        alpha               = np.float64(params.get('alpha'))
        beta_array          = np.arange(1, params['max_beta']+1, 1)
        self.Lidx           = params.get('Lidx')
        Rf0                 = np.float64(params.get('Rf0'))
        Rm                  = params.get('Rm')
        self.Rm             = np.array(Rm) if isinstance(Rm, list) else Rm*np.ones(len(self.Lidx))
        dt_model            = params.get('dt_model')
        self.optimizer      = params.get('optimizer')
        self.opt_options    = params.get('opt_options')
        self.NP             = params.get('num_pars')
        self.D              = params.get('num_dims')
        self.var_bounds     = params.get('bounds')[:self.D]
        self.par_bounds     = params.get('bounds')[self.D:]
        self.notLidx        = np.array([x for x in list(range(self.D)) if x not in self.Lidx])

        assert(type(dt_model) is int)
        self.dt_model = self.dt_data/dt_model
        self.N_model = (self.N_data - 1) * dt_model + 1
        self.t_model = np.linspace(self.t_data[0], self.t_data[-1], self.N_model)
        if self.stim is not None:
            self.stim = np.interp(self.t_model, self.t_data, self.stim)

        self.model_skip = dt_model
        self.Rf = self._gen_rf(Rf0, alpha, beta_array)


        P0 = np.zeros(self.NP)
        for i, b in enumerate(self.par_bounds):
            P0[i] = self.rng.uniform(low = b[0], high = b[1])

        X0 = self._get_X0(P0)

        self.minpaths = np.concatenate((X0.flatten(), P0))
        self.min_A_arr = np.zeros(len(beta_array))

        lower_var = np.tile(self.var_bounds[:, 0], self.N_model)
        upper_var = np.tile(self.var_bounds[:, 1], self.N_model)
        lower_par = self.par_bounds[:, 0]
        upper_par = self.par_bounds[:, 1]
        self.bounds = np.array(list(zip(np.concatenate((lower_var, lower_par)), np.concatenate((upper_var, upper_par)))))

    def set_data_fromfile(self, data_file, stim_file=None, nstart=0, N=None):
        """Load data & stimulus time series from file.

        If data is a text file, must be in multi-column format with L+1 columns:
            t  y_1  y_2  ...  y_L
        If a .npy archive, should contain an Nx(L+1) array with times in the
        zeroth element of each entry.
        Column/array formats should also be in the form t  s_1  s_2 ...
        """

        data = np.load(data_file) if data_file.endswith('npy') else np.loadtxt(data_file)

        self.N_data = N if N is not None else data.shape[0]
        self.t_data = data[nstart:nstart+self.N_data, 0]
        self.dt_data = self.t_data[1] - self.t_data[0]
        self.Y = data[nstart:nstart+self.N_data, 1:]

        if stim_file is not None:
            self.stim = np.load(stim_file)[nstart:nstart+self.N_data, 1:] if stim_file.endswith('npy')\
                                           else np.loadtxt(stim_file)[nstart:nstart+self.N_data, 1:]
            self.stim = np.squeeze(self.stim.T)

        else: self.stim = None


    def min_A(self, id):
        '''Routine to run DA routine
        Args:
            id for saving
        '''
        #print data to file for debugging
        with open(self.data_folder+self.name+'_{:d}.txt'.format(id), 'bw+') as file_temp:
            for i, rf in enumerate(self.Rf):
                self.rf = rf
                self.minpaths, self.min_A_arr[i] = self._min_A_step(i)
                print('iteration: {:d}'.format(i))
                print('param est: ', self.minpaths[self.D*self.N_model:])
                print('Action level: ', self.min_A_arr[i])
                np.savetxt(file_temp, self.minpaths.reshape(1, -1), fmt = '%1.5e')
                file_temp.flush()

        #remove when run sucessfully
        os.remove(self.data_folder+self.name+'_{:d}.txt'.format(id))
        return self.minpaths[:self.D*self.N_model], self.minpaths[self.D*self.N_model:], self.min_A_arr, self.t_model

    ############# PRIVATE FUNCTIONS #############

    def _get_X0(self, P0):
        stim = self.stim if self.stim is not None else np.empty(self.N_model)
        f = lambda r, t, params: np.array(self.f(r, t, *params))
        X0 = np.zeros((self.N_model, self.D))
        for i, b in enumerate(self.var_bounds):
            X0[:, i] = self.rng.uniform(low =b[0], high = b[1], size = self.N_model)
        X0[::self.model_skip, self.Lidx] = self.Y
        for k in range(self.N_model-1):
            X0[k+1,self.notLidx]  = RK4(f, X0[k], self.t_model[k], self.dt_model,
                                        params = (P0, stim[i-1]))[self.notLidx]
        X0[::self.model_skip, self.Lidx] = self.Y
        return X0

    def _gen_rf(self, Rf0, alpha, beta):
        Rf0 = np.array(Rf0) if isinstance(Rf0, list) else Rf0*np.ones(self.D)
        alpha = np.array(alpha) if isinstance(alpha, list) else alpha*np.ones(self.D)
        assert(len(alpha) == self.D and len(alpha) == len(Rf0))
        for b in beta:
            yield Rf0*alpha**b

    '''
    Compute the action.  Function for pyoptsparse
    '''
    def _action(self, x):
        X = x[:self.N_model*self.D]
        p = x[self.N_model*self.D:]

        X = np.reshape(X, (self.N_model, self.D))

        diff_m = X[::self.model_skip, self.Lidx] - self.Y
        merr = np.linalg.norm(np.sqrt(self.Rm)*diff_m)**2
        merr/=(len(self.Lidx) * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(self.f, X, p, self.stim, self.t_model)
        ferr = np.linalg.norm(np.sqrt(self.rf)*diff_f)**2
        ferr/=(self.D * (self.N_model - 1))

        return merr+ferr

    '''
    Compute gradient of the action.  Function for pyoptsparse
    '''
    def _grad_action(self, x):
        X = x[:self.N_model*self.D]
        p = x[self.N_model*self.D:]

        X = np.reshape(X, (self.N_model, self.D))

        diff_m = X[::self.model_skip, self.Lidx] - self.Y

        dmdx = np.zeros((self.N_model, self.D))
        dmdx[::self.model_skip, self.Lidx] = self.Rm*diff_m
        dmdx = dmdx.flatten()
        dmdx/=(len(self.Lidx)/2 * self.N_data)

        diff_f = X[1:] - self.disc_trapezoid(self.f, X, p, self.stim, self.t_model)

        dfdx = self._get_dfdx(X, p, self.fjacx, self.N_model, self.D,
                              self.t_model, self.rf, diff_f, self.stim)

        dfdp = self._get_dfdp(X, p, self.fjacp, self.N_model, self.D,
                              self.t_model, self.rf, diff_f, self.stim)

        return np.concatenate((dmdx+dfdx, dfdp))

    def _optimize(self, XP0):
        if self.optimizer.upper() == 'CYIPOPT':
            res = cyipopt.minimize_ipopt( self._action,
                                        XP0,
                                        jac=self._grad_action,
                                        bounds=self.bounds,
                                        options=self.opt_options)
            xstar = res.get('x')

        elif self.optimizer.upper() == 'SNOPT':
            m = 1
            n = len(self.bounds[:, 0])
            x0 = np.zeros(m+n)
            x0[:n] = XP0
            J = np.zeros(n).reshape(1, -1)
            J[0] = 100
            bl    = -np.inf*np.ones(n+m)
            bl[:n]=  self.bounds[:, 0]
            bu    =  np.inf*np.ones(n+m)
            bu[:n]=  self.bounds[:, 1]

            res = snoptb(self._snopt_obj,
                         self._snopt_con,
                         nnObj = n,
                         nnCon = 0,
                         nnJac = 0,
                         iObj = 0,
                         x0 = x0,
                         bl = bl,
                         bu = bu,
                         J = J,
                         name = 'action',
                         options = self.opt_options)
            xstar = res.x[:-1]
        return xstar, self._action(xstar)

    def _min_A_step(self, beta_i):
        XPmin, Amin = self._optimize(self.minpaths)
        return XPmin, Amin


    @staticmethod
    def _snopt_con(mode,x,fCon,gCon,nState):
        # No nonlinear constraints
        return mode, fCon, gCon


    def _snopt_obj(self, mode, x, fObj, gObj, nState):
        fObj = 0.0
        if mode == 0 or mode == 2:
            fObj = self._action(x)

        if mode == 0:
            return mode, fObj

        if mode == 1 or mode == 2:
            gObj[:] = self._grad_action(x)

        return mode, fObj, gObj

    @staticmethod
    @njit
    def disc_trapezoid(f, x, p, stim, t_model):
        """
        Time discretization for the action using the trapezoid rule.
        """
        N_model, D = x.shape
        dt_model = t_model[1] - t_model[0]
        fn = np.zeros((N_model-1, D))
        fnp1 = np.zeros((N_model-1, D))
        if stim is None: stim = np.empty(N_model)

        for i in range(N_model):
            eval_f = np.array(f(x[i], t_model[i], p, stim[i]))
            if i != N_model-1:
                fn[i] = eval_f
            if i != 0:
                fnp1[i-1] = eval_f

        return x[:-1] + dt_model * (fn + fnp1) / 2


    '''
    Helper functions for computing the action gradient
    '''

    @staticmethod
    @njit
    def _get_dfdx(X, p, fjacx, N_model, D, t_model, rf, diff_f, stim):
        if stim is None: stim = np.empty(N_model)
        dt_model = t_model[1] - t_model[0]
        dfdx = np.zeros((N_model, D))

        J = fjacx(X[0], t_model[0], p, stim[0])
        dfdx[0] = np.sum(-(rf*diff_f[0]).reshape(-1, 1)*(np.eye(D) + 0.5*dt_model*J),
                             axis = 0)

        for i in range(1, N_model-1):
            J = fjacx(X[i], t_model[i], p, stim[i])
            df1 = np.sum((rf*diff_f[i-1]).reshape(-1, 1)*(np.eye(D) - 0.5*dt_model*J), axis = 0)
            df2 = np.sum(-(rf*diff_f[i]).reshape(-1, 1)*(np.eye(D) + 0.5*dt_model*J), axis = 0)
            dfdx[i] = (df1 + df2)

        J = fjacx(X[-1], t_model[-1], p, stim[-1])
        dfdx[-1] = np.sum((rf*diff_f[-1]).reshape(-1, 1)*(np.eye(D) - 0.5*dt_model*J),
                             axis = 0)
        return (2*dfdx/(D * (N_model - 1))).flatten()

    @staticmethod
    @njit
    def _get_dfdp(X, p, fjacp, N_model, D, t_model, rf, diff_f, stim):
        if stim is None: stim = np.empty(N_model)
        dfdp = np.zeros((N_model, len(p)))
        dt_model = t_model[1] - t_model[0]

        G = fjacp(X[0], t_model[0], p, stim[0])
        for i in range(N_model-1):
            Gp1 = fjacp(X[i+1], t_model[i+1], p, stim[i+1])
            dfdp[i] = np.sum((rf*diff_f[i]).reshape(-1, 1)*(G+Gp1), axis = 0)
            G = Gp1
        dfdp = -dt_model*np.sum(dfdp, axis = 0)/(D * (N_model - 1))
        return dfdp