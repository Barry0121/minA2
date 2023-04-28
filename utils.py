import numpy as np
import yaml


def RK4(f, r, t, dt, params = None):
    k1 = dt*f(r,t, params)
    k2 = dt*f(r+k1/2, t+dt/2, params)
    k3 = dt*f(r+k2/2, t+dt/2, params)
    k4 = dt*f(r+k3, t+dt, params)
    return r + (k1+2*k2+2*k3+k4)/6

def RK6(f, r, t, dt, params=None):
    k1 = dt * f(r, t, params)
    k2 = dt * f(r + k1 / 4, t + dt / 4, params)
    k3 = dt * f(r + k1 / 8 + k2 / 8, t + dt / 4, params)
    k4 = dt * f(r - k2 / 2 + k3, t + dt / 2, params)
    k5 = dt * f(r + 3 * k1 / 16 + 9 * k4 / 16, t + 3 * dt / 4, params)
    k6 = dt * f(r - 3 * k1 / 7 + 2 * k2 / 7 + 12 * k3 / 7 - 12 * k4 / 7 + 8 * k5 / 7, t + dt, params)

    return r + (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) / 90

# def RK7(f, r, t, dt, params=None):
#     a = [
#         [1/6],
#         [4/75, 16/75],
#         [5/6, -8/3, 5/2],
#         [-165/64, 55/6, -425/64, 85/96],
#         [12/5, -8, 4015/612, -11/36, 88/255],
#         [-8263/15000, 124/75, -643/680, -81/250, 2484/10625],
#         [3501/1720, -300/43, 297275/52632, -319/2322, 24068/84065, 3850/26703]
#     ]
#     b = [3/40, 0, 875/2244, 23/72, 264/1955, 125/11792, 43/6, 1/20]

#     k = [None] * 8
#     k[0] = dt * f(r, t, params)
#     for i in range(1, 8):
#         args = r + sum(a[i-1][j] * k[j] for j in range(i))
#         k[i] = dt * f(args, t + dt * a[i-1][i-1], params)

#     return r + sum(b[i] * k[i] for i in range(8))

def read_specs(path_to_specs):
    with open(path_to_specs) as file:
        specs = yaml.load(file, Loader=yaml.FullLoader)
    return specs

def read_bounds(path_to_bounds):
    return np.loadtxt(path_to_bounds, delimiter=',')

