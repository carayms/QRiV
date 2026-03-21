import os.path
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['font.family'] = ['DejaVu Sans', 'Arial'][1]
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, expit
from sklearn.model_selection import KFold
from scipy.optimize import minimize, root
import scipy.special as spsp
import scipy.stats as sps
#from numba import jit

import realDataRuncs as Funcs
import importlib
importlib.reload(Funcs)
import pickle

def quantile_loss_rdt(params, x, y, tau, gtype = 'none'):
    if gtype == 'none':
        raise ValueError(f"[gtype] must be specified, not {gtype}")
    residuals = y - rdt_mfunc(params, x, gtype)
    rtn = np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))
    return rtn

def rdt_mfunc(prs, cvats, gtype = 'cherry'):
    if gtype == 'cherry':
        mfc = prs[0]*100 + prs[1]*cvats.T[0] + \
              prs[2]*np.log(cvats.T[1])*10 + \
              prs[3]*cvats.T[2] + prs[4]*cvats.T[3] + prs[5]*cvats.T[4]
    else:
        raise ValueError(f"[gtype] must be specified, not {gtype}")
    return mfc

def gener_mfunc_v2(prs, x_ls, ers, gtype = 'none'):
    rtn = rdt_mfunc(prs, x_ls, gtype) + ers
    return rtn


def est_eqs_loop_jstcal_v2(prs, h_ker, Y, Ws, u_std, Quadts, qtl, bds, gtype = 'none'):
    if gtype == 'none':
        raise ValueError(f"[gtype] must be specified, not {gtype}")

    for iprs, (ia, ib) in zip(prs, bds):
        if (iprs - ia) * (iprs - ib) > 0:
            rtn = np.ones(prs.size) * 9999999
            return rtn

    # --- following is calculation in matrix from, not for loop
    tj, wtj = Quadts
    body_V1 = 1j * u_std[0] * tj
    body_V2 = 1j * u_std[1] * tj
    wts = np.outer(wtj, wtj)

    outer_eqs = np.zeros([len(Ws), prs.size])
    for idx_o, O_i in enumerate(Ws):
        W1, W2 = O_i[:2]

        extrop_X1 = W1 + body_V1
        extrop_X2 = W2 + body_V2
        varib_X2 =  np.log(extrop_X2)*10

        SQ = np.add.outer(prs[1]*extrop_X1, prs[2]*varib_X2, dtype = np.complex128)

        mfunc = prs[0]*100 + SQ + prs[3]*O_i[2] +prs[4]*O_i[3]+prs[5]*O_i[4]

        phi_in = Y[idx_o] - mfunc
        phi_in_scl = phi_in/h_ker

        phi_out = sps.norm.cdf(phi_in_scl, loc = 0, scale = 1) + qtl - 1
        phi_out += phi_in * sps.norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker

        eq1s = phi_out
        eq2s = phi_out * np.expand_dims(extrop_X1, axis = 1)
        eq3s = phi_out * np.expand_dims(varib_X2, axis = 0)
        eq4s = phi_out * O_i[2]
        eq5s = phi_out * O_i[3]
        eq6s = phi_out * O_i[4]

        eq_forming = np.array([eq1s, eq2s, eq3s, eq4s, eq5s, eq6s]) * wts
        outer_eqs[idx_o] = np.sum(eq_forming.real, axis = (1, 2))

    rtns = np.mean(outer_eqs, axis = 0)
    return rtns

if __name__=="__main__":
    n_repts = 1050
    n_simex = 20
    k_kfold = 10
    n = 1000
    
    mdl_type = 'cherry'
    prs_truth = np.array([1.5, -0.7, -3.2, -0.1, 0.7, -0.1])
    n_prs = prs_truth.size
    
    lam_step = 0.25
    lambda_add = np.arange(0, 2.1, lam_step)

    n_lambda = lambda_add.size
    v_a, v_b = [2.0, -0.1]

    run_type = 'hpc..'

    if run_type == 'hpc':
        
        tau = float(sys.argv[1])
        tol = float(sys.argv[2])
        tunning_c = np.array(str(sys.argv[3]).split(',')).astype(float)
        npsd_data = int(sys.argv[4])
    else:
        tau = 0.5 
        tol = 1e-7
        tunning_c = np.array([13])
        npsd_data = 1652

    n_tunning = tunning_c.size
    tunning_string = ';'.join(tunning_c.astype(str))
    tol_hybr = tol

    write_flag = [None, False, True][2]

    # Quadrature: Gauss-Hermit normally distributed V.
    tdeg = 2
    tGH, wt_tGH = spsp.roots_hermitenorm(tdeg)
    quadratures = Vi, WTi = tGH, wt_tGH/np.sqrt(2*np.pi)
    h_ord = -1.0/3
    h_dict = {-1/2: '-1o2', -1/3: '-1o3', -2/5:'-2o5'}

    # merr_error settings
    merr_mean = np.zeros(2)
    merr_covs = np.diag([3.88, 1.85])

    merr_std_all = np.diag(merr_covs)**0.5
  
    # algo setup
    optm_method = ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'TNC'][0]
    bounds = np.tile([-100, 100], n_prs).reshape(n_prs, 2)

    rng = np.random.default_rng(npsd_data)
    rng_seed = rng.choice(np.arange(1000000), size=n_repts, replace=False)
    ini_seeds = np.random.default_rng(seed=0).choice(np.arange(1000000), size=n_repts, replace=False)
    # ---- bootstrap and inin
    npsd_bstrp = 99
    org_idx = np.arange(n) # this is needed for btsrp
    rng_bstrp_seed = np.random.default_rng(seed=npsd_bstrp).choice(1000000, size=n_repts, replace=False)
    
    npsd_merr = 1999
    npsd_eps = 1655
    npsd_cvali = 228
    eps_seed = np.random.default_rng(seed=npsd_eps).choice(np.arange(1000000), size=n_repts, replace=False)
    mer_seed = np.random.default_rng(seed=npsd_merr).choice(np.arange(1000000), size=n_repts, replace=False)
    cvali_seed = np.random.default_rng(seed=npsd_cvali).choice(np.arange(1000000), size=n_repts, replace=False)


    #------ creat the output files destination
    output_dir = f'SimEx_bandwidthSelc_tau.{tau}_tol.{tol:.0e}_dtnpsd.{npsd_data}_n.{n}_mdlErr.{v_a}n{v_b}'
    dump_file = f"{output_dir}/SiMex_tau.{tau}_tun.{tunning_string}_crosvali.{k_kfold}-fd_n.rept.{n_simex}.pkl"

    if not os.path.exists(output_dir):
        if write_flag:
            try:
                os.mkdir(output_dir)
            except:
                pass
        else:
            print('No output written!!')
    else:
        sys.stderr.write('\n%s is existed, keep going!\n' % output_dir)


    loss_ls = np.zeros([n_simex, n_lambda, n_tunning, k_kfold])
    cov = np.array([[1.0, -0.2, 0.0],
                    [-0.2, 1, 0.7],
                    [0.0, 0.7, 1]])


    '''
    start the simex
    '''
    for idx_rpts in range(n_simex):#
        ini_prs = np.random.default_rng(seed=ini_seeds[idx_rpts]).normal(0, 1, n_prs)
        rng_mer = np.random.default_rng(mer_seed[idx_rpts])

        #------
        rng_one = np.random.default_rng(rng_seed[idx_rpts])
        rng_eps = np.random.default_rng(eps_seed[idx_rpts])

        # generate the data
        unobvrX2 =  rng_one.normal(loc = 60, scale = 13, size = n)
        unobvrX2 = np.clip(unobvrX2, 0, np.inf)
        
        gamma_samples = rng_one.gamma(shape=0.56, scale=1/0.06, size=n)
        obsvrZ1 = rng_one.binomial(n=1, p=.6, size=n)*gamma_samples

        # copula structure
        gauss_copula = rng_one.multivariate_normal(np.zeros(3), cov, size = n)
        unobvrX1 = 6 + 3*gauss_copula.T[0]
        unobvrX = np.column_stack([unobvrX1, unobvrX2])

        obsvrZ23 = np.column_stack([36 + 2*gauss_copula.T[1],
                                    130 + 14*sps.norm.cdf(gauss_copula.T[2])])

        # centering
        unobvrX[:, 0] -= unobvrX[:, 0].mean()
        obsvrZ1 -= obsvrZ1.mean()
        obsvrZ23 -= obsvrZ23.mean(axis = 0)

        unobvrs = np.column_stack([unobvrX, obsvrZ1, obsvrZ23])


        # noise
        noise_std = v_a + unobvrs.T[1]**v_b
        eps_nosft = rng_eps.normal(loc = 0, scale = noise_std, size = n)
        eps = eps_nosft - sps.norm.ppf(tau, loc = 0, scale = noise_std)

        respo = gener_mfunc_v2(prs_truth, unobvrs, eps, gtype = mdl_type)

        # genrate observed
        obs_errs = rng_mer.multivariate_normal(merr_mean, merr_covs, size = n)
        obs_ws_base = obs_errs + unobvrX
        obs_ws_base[:, 1] = np.clip(obs_ws_base[:, 1], 0, np.inf)

        # simex prepare
        obvrs_list = []
        for idx_lam,lam in enumerate(lambda_add):

            obs_errs = rng_mer.multivariate_normal(merr_mean, merr_covs, size = n)
            obs_ws_ith = np.sqrt(lam)*obs_errs + obs_ws_base
            obs_ws_ith[:, 1] = np.clip(obs_ws_ith[:, 1], 0, np.inf)
            obvrs_ith = np.column_stack([obs_ws_ith, obsvrZ1, obsvrZ23])
            obvrs_list.append(obvrs_ith)


        rng_cv = np.random.default_rng(cvali_seed[idx_rpts])
        rdm_state = rng_cv.integers(low=999999)
        print(f"\nrept_kfold: {idx_rpts}/{n_simex}; rdm_state: {rdm_state} [for rdm errors]")
        rkf = KFold(n_splits=k_kfold, shuffle=True, random_state=rdm_state)

        kfold_loss = np.zeros([n_lambda, n_tunning, k_kfold]) 
        print('Fold ')
        for idx_fold, (train_index, test_index) in enumerate(rkf.split(respo)):
            print(idx_fold, end = ', ')

            for idx_obvrs, obvrs in enumerate(obvrs_list):
                training_obvrs = obvrs[train_index]
                training_respo = respo[train_index]
                n_training = training_respo.size

                if idx_obvrs == 0:
                    testing_obvrs = unobvrs[test_index]
                else:
                    testing_obvrs = obvrs_list[idx_obvrs-1][test_index]

                testing_respo = respo[test_index]
                # for each split, we do tunning hyperparameter c
                naive_res = minimize(quantile_loss_rdt, ini_prs, args=(training_obvrs, training_respo, tau, mdl_type),method=optm_method,
                                 tol = tol, bounds = bounds)
                # smoothe
                naive_eps = training_respo - gener_mfunc_v2(naive_res.x, training_obvrs, 0, gtype = mdl_type)
                h_base = np.std(naive_eps, ddof = n_prs) * np.power(n_training, h_ord)

                for idx_c, hyper_c in enumerate(tunning_c):
                    h_kern = hyper_c * h_base
                   
                    if idx_obvrs == 0:
                        propo_res = root(est_eqs_loop_jstcal_v2, x0 = naive_res.x,
                                  method='hybr', tol = tol_hybr, args = (h_kern, training_respo, training_obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))
                    else:
                        propo_res = root(est_eqs_loop_jstcal_v2, x0 = naive_res.x,
                                  method='hybr', tol = tol_hybr, args = (h_kern, training_respo, training_obvrs, merr_std_all*np.sqrt(lam_step), quadratures, tau, bounds, mdl_type))

                    beta_hat_h = propo_res.x
                    testing_loss = quantile_loss_rdt(beta_hat_h, testing_obvrs, testing_respo, tau, gtype = mdl_type)

                    kfold_loss[idx_obvrs][idx_c][idx_fold] = testing_loss
        loss_ls[idx_rpts] = kfold_loss

    with open(dump_file, 'wb') as f:
        pickle.dump(loss_ls, f)
    print(f'check _crosvali SimEx: {n_simex} has been processed..')





