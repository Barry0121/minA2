import numpy as np

u0 = [-80.0, 1.0, 0.0, 0.322, 0.7]
p = (450.0, 50.0, 2.0, 3.0, 10.0, 2.0)

def sig(x, y, z):
    return 0.5 * (1.0 + np.tanh(-(x - y) / (2.0 * z)))

def E(x, y, z):
    return np.exp(-(x - y) / z)

def I_stim(t):
    return

def dynamics(du, u, p, t):
    # Constants
    tauNbar, tauH, taur0, taur1, thrT = 10.0, 1.0, 40.0, 17.5, 68.0

    ENa, EK, EL, ECa = 45.0, -90.0, -70.0, 85.0

    thetaM, thetaN, thetaH, sigmaM, sigmaN, sigmaH, thetaS, sigmaS = -35.0, -30.0, -37.4, -5.0, -5.0, 4.3, -30.0, -12.0

    thetaaT, sigmaaT, thetabT, sigmabT, thetarT, sigmarT = -65.0, -7.8, .4, -.1, -68.0, 2.2

    sgmrt, phirT, thetaRF, sigmaRF, thetaRS = 2.2, 1.0, -105.0, 5.0, -105.0

    sigmaRS, kr, ks, Cm = 25.0, .35, .35, 100.0

    f, eps, kca = .1, .0015, .3

    v, h, n, ca, rT = u
    gNa, gK, gSK, gCaT, gCaL, gL = p

    # Na+ and K+ Equations and Currents
    minf = sig(v, thetaM, sigmaM)
    ninf = sig(v, thetaN, sigmaN)
    hinf = sig(v, thetaH, sigmaH)

    tauN = tauNbar / np.cosh((v - thetaN) / (2 * sigmaN))

    iNa = gNa * (minf ** 3) * h * (v - ENa)
    iK = gK * (n ** 4) * (v - EK)

    # L-Type Ca++ Equations and Current
    sinf = sig(v, thetaS, sigmaS)
    iCaL = gCaL * (sinf ** 2) * (v - ECa)

    # T-Type Ca++ Equations and Current
    aTinf = sig(v, thetaaT, sigmaaT)
    bTinf = sig(rT, thetabT, sigmabT) - sig(0, thetabT, sigmabT)
    iCaT = gCaT * (aTinf ** 3) * (bTinf ** 3) * (v - ECa)

    rTinf = sig(v, thetarT, sigmarT)
    taurT = taur0 + taur1 * sig(v, thrT, sgmrt)

    # SK Equations and Current
    kinf = (ca ** 4) / (ca ** 4 + ks ** 4)
    iSK = gSK * kinf * (v - EK)

    # Leak current
    iL = gL * (v - EL)

    # Update du
    du[0] = (-iNa - iK - iCaL - iCaT - iSK - iL + I_stim(t)) / Cm
    du[1] = (hinf - h) / tauH
    du[2] = (ninf - n) / tauN
    du[3] = -f * (eps * (iCaL + iCaT) + kca * (ca - 0.1))
    du[4] = phirT * (rTinf - rT) / taurT
"""
The first value is from the first script, and the second value is from the second script (Jason's code). I've marked the different values with an asterisk (*):

Cm: 100pF vs 100.0
g_Na: 450nS vs 450.0
g_K: 50nS vs 50.0
g_CaL: 19nS vs 10.0*
g_CaT: 2.7nS vs 2.0
g_SK: 6nS vs 3.0*
g_L: 2nS vs 2.0
E_Na: 50mV vs 45.0*
E_K: -90mV vs -90.0
E_L: -70mV vs -70.0
E_Ca: 0, 90 vs 85.0*
theta_aT: -65mV vs -65.0
sigma_aT: -7.8mV vs -7.8
theta_bT: 0.4mV vs 0.4
sigma_bT: -0.1mV vs -0.1
theta_rT: -67mV vs -68.0*
sigma_rT: 2mV vs 2.2*
theta_m: -35mV vs -35.0
sigma_m: -5mV vs -5.0
theta_s: -20mV vs -30.0*
sigma_s: -0.05mV vs -12.0*
theta_n: -30mV vs -30.0
sigma_n: -5mV vs -5.0
tau_r0: 200ms vs 40.0*
tau_r1: 87.5ms vs 17.5*
k_Ca: 0.3ms^-1 vs 0.3
ks: 0.5μM vs 0.35*
f: 0.1 vs 0.1
th: 1ms vs Not found in Jason's code
tn: 10ms vs Not found in Jason's code
Parameters present in the first script but not found in Jason's code: th and tn.

The following parameters have different values between the two scripts: g_CaL, g_SK, E_Na, E_Ca, theta_rT, sigma_rT, theta_s, sigma_s, tau_r0, tau_r1, and ks."""