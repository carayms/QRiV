import numpy as np
import numba as nb
from numba import njit
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from sklearn.linear_model import QuantileRegressor

from sklearn.utils.fixes import parse_version, sp_version
# This is line is to avoid incompatibility if older SciPy version.
# You should use `solver="highs"` with recent version of SciPy.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

from sklearn import datasets, linear_model
from scipy import integrate

import scipy.stats as sps
from scipy.optimize import root

import math
plw = 1.

def bestring(ary):
    rtn = []
    for each in ary:
        str_trans = '$'.join( each.round(3).astype(str) )
        rtn.append( str_trans )
    return rtn

def trans(val, power):
    if power != 0:
        tval = (val ** power - 1) / power
    else:
        tval = np.log(np.maximum(val, 1e-10))
    return tval

def gener_mfunc_v2(prs, x_ls, ers, gtype = 'none'):
    if gtype == 'linear':
        if type(ers) == int and ers == 0:
            rtn = np.sum(prs*x_ls, axis = 1)
        else:
            rtn = np.sum(prs*x_ls, axis = 1) + ers
    elif gtype == 'square':
        rtn = prs[0] + prs[1] * x_ls.T[1]**2 + ers
    elif gtype == 'trigonometric':
        rtn = prs[0] + prs[1]*np.sin(prs[2] * x_ls.T[1]) + ers
    else:
        input('m(x) not found!')
    return rtn

def approx_step_func(epsilon, bw_h, smooth_name = 'normcdf'):
    """
    smooth_name is a selection of a sigmod function, which includes any S-shaped functions. 
    """
    eps_scld = epsilon/bw_h
    
    if smooth_name == 'normcdf':
        rtn = sps.norm.cdf(eps_scld, loc = 0, scale = 1)
    elif smooth_name == 'logistic':
        rtn = 1.0/(1+np.exp(-eps_scld))
    elif smooth_name == 'algebric':
        rtn = 0.5 + 0.5*eps_scld/np.sqrt(1 + eps_scld**2)

    else:
        input('smooth_name not found!')

    return rtn

def approx_step_func_deri(epsilon, bw_h, smooth_name = 'normcdf'):
    eps_scld = epsilon/bw_h
    
    if smooth_name == 'normcdf':
        rtn = sps.norm.pdf(eps_scld, loc = 0, scale = 1) / bw_h

    elif smooth_name == 'logistic':
        fval = 1.0/(1+np.exp(-eps_scld))
        rtn = fval * (1 - fval) /bw_h
    else:
        input('smooth_name not found!')

    return rtn

def rho_prime_smooth_v2(params, x, y, tau, bandwth, xmat_shape, gtype = 'linear', smtype = 'normcdf'):
    #print(gtype)
    y_hat = gener_mfunc_v2(params, x, 0, gtype = gtype)
    residuals = y - y_hat
    psi = tau - 1 + approx_step_func(residuals, bandwth, smtype)
    psi += residuals * approx_step_func_deri(residuals, bandwth, smtype)
    deris = deri_func(x, xmat_shape, gtype = gtype, prs = params)
    rtn = np.expand_dims(psi, axis = 1) * deris
    rtn = -rtn.mean(axis = 0)
    return rtn

def deri_func(x_ls, xmat_shape, gtype = 'linear', prs = None):
    if gtype == 'linear':
        rtn = x_ls
    elif gtype == 'square':
        rtn = np.ones(xmat_shape)
        rtn[:, 1] = x_ls.T[1]**2

    elif gtype == 'trigonometric':
        rtn = np.ones(xmat_shape)
        rtn[:, 1] = np.sin(prs[2] * x_ls.T[1])
        rtn[:, 2] = prs[1]*prs[2]*np.cos(prs[2] * x_ls.T[1])
    else:
        input('m(x) not found!')

    return -rtn

# Define the quantile loss function
def quantile_loss_v2(params, x, y, tau, gtype = 'linear'):
    residuals = y - gener_mfunc_v2(params, x, 0, gtype = gtype)
    return np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))

def corected_qloss(prs, Y, covrits, u_std, qtl, h_krl, treturn = 'meanLoss'):

    mfuc_w = gener_mfunc_v2(prs, covrits, 0, gtype = 'linear')
    epsi_w = Y - mfuc_w
    epsi_w_srtidx = np.argsort(epsi_w)

    cst_part = epsi_w*(qtl - 0.5)

    y_grid = np.arange(1e-5, 1/h_krl+1e-5, v_step)#.reshape(1, n_of_y)
    n_of_y = y_grid.size
    y_eps = np.outer(epsi_w, y_grid)

    #input(y_eps.shape)
    u_var = (prs[1]*u_std)**2    
    ingrd = (1/y_grid * np.expand_dims(epsi_w, axis = 1) * np.sin(y_eps) - u_var*np.cos(y_eps)) * np.exp(y_grid**2 * u_var * 0.5)/np.pi

    A_func_val = cst_part + ingrd.sum(axis = 1) * v_step

    if treturn == 'meanLoss':
        rtns = np.mean(A_func_val)
    else:
        rtns = A_func_val

    return rtns

def eps_std(coefs, x_ls, dep_fun = 'linear'):
    a_, b_ = coefs
    if dep_fun == 'linear':
        rtn = a_ + b_ * x_ls
    elif dep_fun == 'expo':
        rtn = np.exp(a_ + b_ * x_ls)
    elif dep_fun == 'expo_decay':
        rtn = a_ + np.exp( -b_ * x_ls)
    elif dep_fun == 'square':
        rtn = a_ + b_ * np.square(x_ls)
    elif dep_fun == 'trigtri':
        rtn = a_ + 0.3*np.absolute(np.sin(b_*x_ls))
    #print(dep_fun)

    dep_fun = 'linear'
    return rtn

@nb.jit(nopython=True)
def aprx_norm_cdf(vrbs):
    #tmp = 1.526 * vrbs * (1 + 0.1034 * vrbs) # Divgi (1990) 2.10 × 10−3
    tmp = np.sqrt(np.pi)*(0.9*vrbs + 0.0418198 * vrbs**3 - 0.0004406 * vrbs**5) #Waissi and Rossin (1996) 4.37 × 10−5
    rtn = 1/(1 + np.exp(-tmp))
    return rtn

def est_eqs_loop_jstcal_v2(prs, h_ker, Y, Ws, u_std, Quadts, qtl, bds, gtype = 'linear'):
    for iprs, (ia, ib) in zip(prs, bds):
        if (iprs - ia) * (iprs - ib) > 0:
            rtn = np.ones(prs.size) * 99999
            return rtn
    
    if gtype == 'linear':
        W = Ws[:, 1]
        tj, wtj = Quadts
        body_V = 1j * u_std * tj

        extrop_X = np.add.outer(W, body_V)

        mfunc = prs[0] + prs[1] * extrop_X 

        phi_in = np.expand_dims(Y, axis = 1) - mfunc
        phi_in_scl = phi_in/h_ker

        phi_out = sps.norm.cdf(phi_in_scl) + qtl - 1
        phi_out += phi_in * sps.norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker

        eq0s = phi_out 
        eq1s = phi_out * extrop_X 
        eq_forming = np.array([eq0s, eq1s]) * wtj

    
        outer_eqs = eq_forming.sum(axis = 2)

    elif gtype == 'square':
        W = Ws[:, 1]
        tj, wtj = Quadts
        body_V = 1j * u_std * tj

        extrop_X = np.square(np.add.outer(W, body_V))

        mfunc = prs[0] + prs[1] * extrop_X

        phi_in = np.expand_dims(Y, axis = 1) - mfunc
        phi_in_scl = phi_in/h_ker
        #print(phi_in_scl.shape)

        phi_out = sps.norm.cdf(phi_in_scl) + qtl - 1

        phi_out += phi_in * sps.norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker
        eq0s = phi_out 
        eq1s = phi_out * extrop_X 

        eq_forming = np.array([eq0s, eq1s]) * wtj
        outer_eqs = eq_forming.sum(axis = 2)

    elif gtype == 'trigonometric':
        W = Ws[:, 1]
        tj, wtj = Quadts
        body_V = 1j * u_std * tj

        extrop_X = np.add.outer(W, body_V)

        mfunc = prs[0] + prs[1] * np.sin(prs[2] * extrop_X)

        phi_in = np.expand_dims(Y, axis = 1) - mfunc
        phi_in_scl = phi_in/h_ker
        #print(phi_in_scl.shape)

        phi_out = sps.norm.cdf(phi_in_scl) + qtl - 1

        phi_out += phi_in * sps.norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker

        eq0s = phi_out 
    
        eq1s = phi_out * np.sin(prs[2] * extrop_X)
        eq2s = phi_out * prs[1]*prs[2]*np.cos(prs[2] * extrop_X) 
        eq_forming = np.array([eq0s, eq1s, eq2s]) * wtj
        outer_eqs = eq_forming.sum(axis = 2)

    else:
        input('no %s in est_eqs_loop_jstcal_v2 - - - - !!!!' % gtype)

    rtns = np.mean(outer_eqs, axis = 1).real 
    return rtns

if __name__=="__main__":
    print('QRfuncV1')


























    

   
