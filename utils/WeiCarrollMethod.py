# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:50:40 2024

@author: caray li
"""

import scipy.stats as sps
from scipy.optimize import root
from scipy import interpolate
from sklearn.linear_model import QuantileRegressor
import copy
import numpy as np
import bisect
import numba as nb
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def weicarroll_qr(tau_ist, extendtau, obvrs_in, y_in, sdu_in, bds, maxiter = 20, tol = 0.01):
    n = y_in.size
    if obvrs_in.shape[1] == 2:
        w_in = obvrs_in[:, 1]

    if w_in.min() <= 0:
        sft_size = -w_in.min() + 1.0
    else:
        sft_size = 0

    #w_sft = w_in + np.random.normal(0, 0.01, w_in.size) + sft_size
    w_sft = w_in + sft_size

    # estimate est_power
    trans_w, est_power = sps.boxcox(w_sft)
    sd_trans_w = np.std(trans_w, ddof = 1)
    mu_trans_w = np.mean(trans_w)

    # sduhat
    rnum = int(n/5)
    rng_u = np.random.default_rng(seed=21)
    rx = rng_u.normal(loc = w_in.mean(), scale = sdu_in, size=rnum)
    w1 = rng_u.normal(loc = rx, scale = sdu_in, size=rnum)
    w2 = rng_u.normal(loc = rx, scale = sdu_in, size=rnum)
    tw1 = trans(w1, est_power); tw2 = trans(w2, est_power)
    sduhat = np.sqrt(np.var(tw1-tw2, ddof = 1)/2)
    print('est_power',est_power, 'sduhat', sduhat, 'sdu_in', sdu_in)
    #input('see')

    
    naive_inis = []
    for ct in extendtau[1:-1]:
        qr = QuantileRegressor(quantile=ct, alpha=0)
        tmp = qr.fit(w_sft[:, np.newaxis], y_in)
        naive_inis.append( [qr.intercept_] +  qr.coef_.tolist() )

    naive_inis = np.array(naive_inis, dtype = np.float64)
    beta_grid = copy.deepcopy(naive_inis)


    # the x
    sd_trans_x = np.sqrt(np.clip(sd_trans_w ** 2 - sduhat ** 2, 1e-10, np.inf))
    mu_trans_x = mu_trans_w

    niter = 1
    itrdiff = 9999
    lastbeta = copy.deepcopy(beta_grid)

    I = n      #  I is the number of observations 
    J = 200    #  J is the number of the x-grid for numerical intergration

    #
    # prepare
    n_tau = len(extendtau)

    len_limit = n_tau-1


    # d is conditional sd of trans(x) given trans(w)
    d_ini = np.clip(sd_trans_x ** 2 * (1 - (sd_trans_x ** 2 / sd_trans_w ** 2)), 1e-10, np.inf)
    d = np.sqrt(d_ini)

    # m is conditional mean of trans(x) given trans(w)
    m_all = mu_trans_w + (sd_trans_x ** 2 / (sd_trans_w ** 2)) * (trans_w - mu_trans_w)


    # xgrid shold be I x J dim. and it is!
    lo = (np.maximum((m_all - 4 * d) * est_power + 1, 1e-10)) ** (1 / est_power)
    up = ((m_all + 4 * d) * est_power + 1) ** (1 / est_power)
    xgrid = np.linspace(lo, up, J).T
    delta = np.diff(xgrid, axis=1)
    #input(xgrid.shape)
    #input(delta.shape)

    # DEN1
    DEN1_tmp = sps.norm.pdf(trans(xgrid.T, est_power), loc=m_all, scale=d) * (xgrid.T ** (est_power - 1))
    DEN1_all = DEN1_tmp.T

    # DEN2 func
    extendtau_diff = np.diff(extendtau)

    # f_matrix
    Y = np.repeat(y_in, J)
    CF = np.zeros(extendtau.size)
    CF[[0, -1]] = [y_in.min(), y_in.max()] 
    X_ones = np.ones([I*J, 2])
    zeros = np.zeros(I)

    smallest_err = 999
    #smallest_err_beta = lastbeta[0]

    while niter <= maxiter and itrdiff > tol:
        
        # E step, Updating p( x |y, w, beta).
        fmatrix = fxdw_den2_outfor(J, y_in, m_all, CF, beta_grid, delta, extendtau, extendtau_diff, n_tau, xgrid, DEN1_all)

        # normalize 
        temp = (fmatrix[:, :-1] + fmatrix[:, 1:]) / 2.0
        temp *= delta
        dem = np.sum(temp, axis=1)
        #print(dem.shape)
        #plt.hist(dem);plt.show()
        non_zero_idx = ~np.isclose(dem, zeros)
        fmatrix_2 = fmatrix
        fmatrix_2[non_zero_idx] = fmatrix_2[non_zero_idx]  / dem[non_zero_idx, None]
        #print("dem has %s 0's" % (I - non_zero_idx.sum()) )

        # M step, note: if we write the estimation equation in the double-sum form, it is
        #         equivalent to a weighted quantile regression with weights 
        #         p( x |y_i, w_i, beta_previous)
        X = np.ravel(xgrid)
        WTs = np.ravel(fmatrix_2)
        X_ones[:,1] = X
        '''
        print(WTs.min(), WTs.max())
        plt.hist(WTs, range = [0, 1])
        plt.show()
        plt.imshow(fmatrix_2)
        plt.show()
        '''
        min_method = ['L-BFGS-B', 'Nelder-Mead', 'Powell'][0]
        for idx_ct, ct in enumerate(extendtau[1:-1]):
            ini_prs = beta_grid[idx_ct]
            qloss_method = minimize(quantile_loss, ini_prs, args=(X_ones, Y, ct, WTs), method=min_method, tol = 1e-15, bounds = bds)
            qr_res = qloss_method#smoth_method#qloss_method #smoth_method #prime_method#
            #qloss_method
            
            
            beta_grid[idx_ct] = qr_res.x

        itrdiff = np.mean(np.abs(beta_grid - lastbeta))
        #itrdiff = np.sum(np.square(beta_grid - lastbeta))
        print("number of iterations =", niter, '->', itrdiff, end = '\n')


        niter += 1
        lastbeta = copy.deepcopy(beta_grid)

        if itrdiff < smallest_err:
            smallest_err = itrdiff
            smallest_err_beta = lastbeta

    if niter >= maxiter and itrdiff > tol:
        print('--!!smallest_err', smallest_err, 'used!!--')
        beta_grid = smallest_err_beta

    beta_grid[:, 0] = beta_grid.T[0] + sft_size * beta_grid.T[1]

    ftau = interpolate.interp1d(extendtau[1:-1], beta_grid.T, axis = 1)
    beta_weicar = ftau(tau_ist)
    
    print('-----beta_weicar->', beta_weicar, 'niter-> %s; err-> %.5f' % (niter, itrdiff) )
    return beta_weicar

#@nb.jit(nopython=True)
def fxdw_den2(xc, y_i, coeffs, cf_in, tauext, tauext_diff, len_end):
    # updated on 2024-08-21, according to the paper and find "pos" 
    # based on tau and ytau, not y_i and y_hat.

    # current x and design.x on 1.
    cdt = np.array([1, xc], dtype = np.float64)
    # predicted y|{current (1, x)}
    cf_in[1:-1] = np.dot(coeffs, cdt)

    
    # approx tau of y_i at, i.e. tau quantile of y|(current x)
    ytau = np.interp(y_i, cf_in, tauext)
    ypos = bisect_right(tauext, ytau)

    if ypos == len_end:
        den2 = 0.0
    elif ypos == 0:
        den2 = 0.0
    else:
        botm = cf_in[ypos]-cf_in[ypos-1]

        den2 = tauext_diff[0]/ botm
    
    return den2

#@nb.jit(nopython=True)
def fxdw_den2_outfor(J, y_in, m_in, CF_in, coeffs, steps, tauext, tauext_diff, len_end, xgrid, DEN1_all):
    I = y_in.size
    fmatrix = np.zeros((I, J)) 
    for i, yi in enumerate(y_in):
        #yi = y_in[i]
        m = m_in[i] 
        DEN2 = np.array([fxdw_den2(xi, yi, coeffs, CF_in, tauext, tauext_diff, len_end) for xi in xgrid[i, :]])
        div = np.sum((DEN2[:-1] + DEN2[1:]) / 2.0) * steps[i, 0]
        if div == 0.0:
            DEN2 = 0.0#DEN2 / 1e-5
        else:
            DEN2 = DEN2 / div

        DEN1 = DEN1_all[i]
        fmatrix[i, :] = DEN1 * DEN2
    return fmatrix

def fxdw_den2_outfor_v2(J, y_in, m_in, CF_in, coeffs, steps, tauext, tauext_diff, len_end, xgrid, DEN1_all, fmatrix):
    for i, yi in enumerate(y_in):
    
        m = m_in[i] 
        DEN2 = np.array([fxdw_den2(xi, yi, coeffs, CF_in, tauext, tauext_diff, len_end) for xi in xgrid[i, :]])
        DEN1 = DEN1_all[i]
        fmatrix[i, :] = DEN1 * DEN2
    return fmatrix


def trans(val, lam):
    if lam != 0:
        tval = (val ** lam - 1) / lam
    else:
        tval = np.log(np.maximum(val, 1e-10))
    return tval

# Define the quantile loss function
def quantile_loss(params, x, y, tau, weights):
    residuals = y - np.dot(x,params)
    theloss = np.maximum(tau * residuals, (tau - 1) * residuals)
    wted_loss = theloss * np.clip(weights, 1e-10, np.inf)
    return wted_loss.mean()


def weighted_rho_prime_smooth(params, x, y, tau, weights, bds):
    for iprs, (ia, ib) in zip(params, bds):
        if (iprs - ia) * (iprs - ib) > 0:
            rtn = np.ones(params.size) * 99999
            return rtn

    residuals = y - np.dot(x,params)
    bandwth = np.std(residuals, ddof = 1) * y.size ** (-1/3) * 20; 

    psi = tau - 1 + sps.norm.cdf(residuals/bandwth, loc = 0, scale = bandwth)
    wted_psi = psi * weights
    rtn = wted_psi[:, np.newaxis] * x
    
    return rtn.mean(axis = 0)

def weighted_rho_prime(params, x, y, tau, weights):
    residuals = y - np.dot(x,params)
    neg = residuals < 0
    print(neg.sum(), neg.shape)
    input('neg')
    #orgni = np.ones(y.size) * (-tau)
    orgni = np.ones(y.size) * tau

    orgni[neg] = orgni[neg] - 1

    wted_psi = orgni * weights

    rtn = wted_psi[:, np.newaxis] * x
    
    return np.square(rtn).mean()

@nb.jit(nopython=True)
def bisect_right(a, x, lo=0, hi=None, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo







if __name__=='__main__':
    a = 0

