#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''

from sympy import *
import numpy as np
from numba import njit

################# MODIFY #################
'''
Specify the dynamical equations using sympy functions
only.  The return value must be a sympy Matrix.
'''
def dynamics(x, t, p, stim = None):
    '''The dynamical equations of the system
    Args:
        x    : The state variables
        t    : Time
        p    : Parameters
        stim : Optional stimulus

    Returns:
        sympy Matrix containing dx/dt
    '''
    return CaFluorescence

##########################################


def get_dynamics(specs):
    '''Compute the jacobian of the dynamics and compile
        to machine code
    Args:
        specs   : dictionary containing the spec file

    Returns:
        f       : dynamical equations
        fjacx   : df/dx
        fjacp   : df/dp
    '''
    func = CaFluorescence # can change to point to different functions

    num_vars = specs['num_dims']
    num_pars = specs['num_par']

    x = np.array(symbols('x:{:d}'.format(num_vars)))
    p = np.array(symbols('p:{:d}'.format(num_pars)))
    stim = symbols('stim')
    t = symbols('t')

    f = func(x, t, p, stim)
    fjacx = func(x, t, p, stim).jacobian(x)
    fjacp = func(x, t, p, stim).jacobian(p)


    lam_f = lambdify((x, t, p, stim), np.squeeze(f))
    lam_fjacx = lambdify((x, t, p, stim), fjacx)
    lam_fjacp = lambdify((x, t, p, stim), fjacp)

    return njit(lam_f), njit(lam_fjacx), njit(lam_fjacp)



########################## EXAMPLES ##########################
'''
Examples of dynamical systems to use purely for referece.
To use change "func = dynamics" line in get_dynamics to point
to the desired function.
'''

def L96(x, t, p, stim = None):
    D = len(x)
    dxdt = zeros(1, D) # 1xD dimensional vector
    for i in range(D-1):
        dxdt[i] = (x[i+1] - x[i-2])*x[i-1] - x[i]
    dxdt[-1] = (x[0] - x[-3])*x[-2]-x[-1]
    dxdt = dxdt + p[0]*ones(1, D)
    return Matrix([dxdt])


def NaKL(x, t, p, stim = None):
    Cm, g_Na, g_K, g_L, E_Na, E_K, E_L, vm, vh, vn, dvm, dvh, dvn, tm0, tm1, th0, th1, tn0, tn1 = p

    V, m, h, n = x

    dvdt = -1/Cm * (g_Na*m**3*h*(V - E_Na) + g_K*n**4*(V - E_K) +
                    g_L*(V - E_L) - stim)

    xm = 0.5*(1+tanh((V - vm)/dvm))
    tm = tm0 + tm1*(1 - tanh((V - vm)/dvm)**2)
    dmdt = (xm - m)/tm

    xh = 0.5*(1+tanh((V - vh)/dvh))
    th = th0 + th1*(1 - tanh((V - vh)/dvh)**2)
    dhdt = (xh - h)/th

    xn = 0.5*(1+tanh((V - vn)/dvn))
    tn = tn0 + tn1*(1 - tanh((V - vn)/dvn)**2)
    dndt = (xn - n)/tn

    return Matrix([dvdt, dmdt, dhdt, dndt])

def sigmoid(x, y, z):
    return 0.5 * (1 + tanh((y - x) / 2*z))

def CaFluorescence(x, t, p, stim = None):
    # unpack (1st line: currents & voltage; 2nd line: gating variables h,n)
    Cm, g_Na, g_K, g_CaL, g_CaT, g_SK, g_L, E_Na, E_K, E_L, theta_aT, sigma_aT, theta_bT, sigma_bT, theta_rT, sigma_rT, theta_m, sigma_m, theta_s, sigma_s, theta_n, sigma_n, tau_r0, tau_r1, k_Ca, ks, f, th, tn = p

    V, h, n, Ca, r_T = x

    Ca_ext = 2.5 # mM
    m_inf = 1.0 / (1.0 + exp((V - theta_m) / sigma_m))
    s_inf = 1.0 / (1.0 + exp((V - theta_s) / sigma_s))
    h_inf = (0.128 * exp((V + 15) / -18)) / (0.128 * exp((V + 15) / -18) + (4 / (1 + exp((V + 27) /  -5))))
    n_inf = 1.0 / (1.0 + exp((V - theta_n) / sigma_n))

    # currents
    E_CaL = 0 # TODO: Ask Jason for feedback, and change accordingly
    E_CaT = 0 # TODO: Use E_ext formulas for E_CaL and E_CaT
    # I_Na, I_K, I_CaL, I_CaT, I_SK, I_L = None # update by equation
    I_Na = g_Na * m_inf**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_CaL = g_CaL * s_inf**2 * (V - E_CaL)
    I_CaT = g_CaT * sigmoid(V, theta_aT, sigma_aT)**3 * (sigmoid(r_T, theta_bT, sigma_bT) - sigmoid(0, theta_bT, sigma_bT))**3 * (V - E_CaT)
    I_SK  = g_SK * (Ca**4 / (Ca**4 + ks**4)) * (V - E_K)
    I_L = g_L * (V - E_L)

    # voltage
    dvdt = -1/Cm * (I_Na + I_K + I_CaL + I_CaT + I_SK + I_L - stim)

    # gating variables
    dhdt = (h_inf - h) / th
    dndt = (n_inf - n) / tn

    # concentration
    dCadt = -f*(I_CaL + I_CaT) - k_Ca*(Ca - 0.1)

    # rT
    r_Tinf = 1/(1 + exp((V - theta_rT) / sigma_rT))
    tau_rT = tau_r0 + tau_r1 / (1 + exp((V - theta_rT) / sigma_rT))
    drTdt = (r_Tinf - r_T) / tau_rT

    return Matrix([dvdt, dhdt, dndt, dCadt, drTdt])
