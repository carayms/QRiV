import os.path
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
# Force DejaVu Sans (it supports basic Unicode flowers)
mpl.rcParams['font.family'] = ['DejaVu Sans', 'Arial'][1]
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, expit

from scipy.optimize import minimize, root
import scipy.special as spsp
import scipy.stats as sps


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


def bs_std_cvg(orgests, truep, bs_ests, lvl = 0.975):
    bs_std = np.sqrt(np.square(bs_ests - orgests).mean(axis = 0))
    bs_half = sps.norm.ppf(lvl, scale = bs_std)
    bs_CIs = np.array([orgests - bs_half, orgests + bs_half])
    olps = bs_CIs - truep
    CI_flag = np.maximum( -np.sign( olps[0] * olps[1]), 0).astype(int)

    return bs_std, CI_flag

if __name__=="__main__":
    n_repts = 1050

    v_a, v_b = [2.0, -0.1]
    prs_truth = np.array([1.5, -0.7, -3.2, -0.1, 0.7, -0.1])

    run_type = 'hpc..'
    if run_type == 'hpc':
        prl_a = int(sys.argv[1])
        prl_b = int(sys.argv[2])
        tau = float(sys.argv[3])
        tol = float(sys.argv[4])
        tuning = float(sys.argv[5])
        npsd_data = int(sys.argv[6])
    else:
        prl_a = 0
        prl_b = 10
        tau = 0.5
        tol = 1e-7
        tuning = 12
        npsd_data = 1652



    tol_hybr = tol

    mdl_type = ['trigonometric', 'square', 'linear', 'cherry'][-1]
    



    n = 1000
    n_prs = prs_truth.size
    n_bs = 200


    write_flag = [None, False, True][-1]
    bstrp_std_indicator = [True, False][0]

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
    # algo set up
    optm_method = ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'TNC'][0]
    bounds = np.tile([-100, 100], n_prs).reshape(n_prs, 2)


    #--------select one date

    #npsd_data = 1652
    rng = np.random.default_rng(npsd_data)
    rng_seed = rng.choice(np.arange(999999), size=n_repts, replace=False)

    ini_seeds = np.random.default_rng(seed=0).choice(np.arange(1000000),
                                                               size=n_repts, replace=False)
    # ---- bootstrap and inin
    npsd_bstrp = 99
    org_idx = np.arange(n) # this is needed for btsrp
    rng_bstrp_seed = np.random.default_rng(seed=npsd_bstrp).choice(1000000, size=n_repts, replace=False)
    # ---- bootstrap and inin over

    est_shape = [n_repts, n_prs]
    est_arsenal = {'predi_ests': np.zeros(est_shape),
                   'naive_ests': np.zeros(est_shape),
                   'propo_est_navini': np.zeros(est_shape),
                   'propo_est_prdini': np.zeros(est_shape)}

    #input('see')
    npsd_merr = 1999
    npsd_eps = 1655


    eps_seed = np.random.default_rng(seed=npsd_eps).choice(np.arange(1000000),
                                                               size=n_repts, replace=False)
    mer_seed = np.random.default_rng(seed=npsd_merr).choice(np.arange(1000000),
                                                               size=n_repts, replace=False)

    #------ creat the output files destination
    output_dir = '%s_tau%s_tol%.e_dtnpsd.%s_n.%s_bstrp%s.npsd.%s' % ('MultiXZ', tau, tol, npsd_data, n, n_bs, npsd_bstrp)

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

    sim_fd = '%s/NmErr_tau%s_n.%s_h.%s.x%s_seed.%s_Quads.%s_tol.%.e_bstrp.%s' % (output_dir, tau, n, h_dict[h_ord], tuning, npsd_data, tdeg, tol, n_bs)
    if not os.path.exists(sim_fd):
        if write_flag:
            try:
                os.mkdir(sim_fd)
            except:
                print('someone has been made it')

    cov = np.array([[1.0, -0.2, 0.0],
                    [-0.2, 1, 0.7],
                    [0.0, 0.7, 1]])
    sep_chr = '\t'

    for idx_rpts in range(n_repts):#n_repts
        if idx_rpts % 10 == 0:
            print(f"{idx_rpts} processing..")
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


        # 1. the response variavle
        # we can adjust it to be x dependent later
        noise_std = v_a + unobvrs.T[1]**v_b
        eps_nosft = rng_eps.normal(loc = 0, scale = noise_std, size = n)
        eps = eps_nosft - sps.norm.ppf(tau, loc = 0, scale = noise_std)

        respo = gener_mfunc_v2(prs_truth, unobvrs, eps, gtype = mdl_type)#.astype(int)

        # 2. genrate observed
        obs_errs = rng_mer.multivariate_normal(merr_mean, merr_covs, size = n)
        obs_ws = obs_errs + unobvrX#df[predi_covts[:2]].to_numpy()
        obs_ws[:, 1] = np.clip(obs_ws[:, 1], 0, np.inf)
        obvrs = np.column_stack([obs_ws, obsvrZ1, obsvrZ23])



        if (idx_rpts - prl_a) * (idx_rpts - prl_b) > 0:
            #print('no')
            continue
        else:
            pass



        predi_res = minimize(quantile_loss_rdt, ini_prs, args=(unobvrs, respo, tau, mdl_type),method=optm_method,#'Nelder-Mead',
                                 tol = tol, bounds = bounds)
        predi_est = ['%.4f' % i for i in predi_res.x]

        naive_res = minimize(quantile_loss_rdt, ini_prs, args=(obvrs, respo, tau, mdl_type),method=optm_method,
                                 tol = tol, bounds = bounds)
        naive_est = ['%.4f' % i for i in naive_res.x]

         # smoothe
        naive_eps = respo - gener_mfunc_v2(naive_res.x, obvrs, 0, gtype = mdl_type)
        h_base = np.power(n, h_ord) * np.std(naive_eps, ddof = n_prs)
        h_kern = h_base * tuning
        #print(f"h_kern->{h_kern}")

        #propo_ini = [naive_est.x, predi_est.x][0]
        propo_res_navini = root(est_eqs_loop_jstcal_v2, x0 = naive_res.x,#propo_ini*pert_noz[-1],#predi_est.x,#naive_est, #ini_prs,
                          method='hybr', tol = tol_hybr, args = (h_kern, respo, obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))
        propo_res_prdini = root(est_eqs_loop_jstcal_v2, x0 = predi_res.x,#propo_ini*pert_noz[-1],#predi_est.x,#naive_est, #ini_prs,
                          method='hybr', tol = tol_hybr,
                                args = (h_kern, respo, obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))

        propo_est_navini = ['%.4f' % i for i in propo_res_navini.x]
        propo_est_prdini = ['%.4f' % i for i in propo_res_prdini.x]

        est_arsenal['naive_ests'][idx_rpts] = naive_res.x
        est_arsenal['predi_ests'][idx_rpts] = predi_res.x
        est_arsenal['propo_est_navini'][idx_rpts] = propo_res_navini.x
        est_arsenal['propo_est_prdini'][idx_rpts] = propo_res_prdini.x


        if idx_rpts == 0:
            print(f'-----> propo by navini status:{propo_res_navini.status}, success:{propo_res_navini.success}, message:{ propo_res_navini.message}')
            print(f'\n---> fun.loss:{propo_res_navini.fun.round(2)}')



            print(f"tau->{tau}")
            print(f"naive_est{sep_chr}{sep_chr.join(naive_est)}")
            print(f"errfe_est{sep_chr}{sep_chr.join(predi_est)}")
            print(f"propo_est{sep_chr}{sep_chr.join(propo_est_navini)} by navini")

        # bstrp coverage
        proposed_flag = True; proposed_succfail = [0, 0]

        bstrp_propo_res = []
        bstrp_naive_res = []
        bstrp_errfre_res = []



        if bstrp_std_indicator:

            rng_bstrp = np.random.default_rng(seed=rng_bstrp_seed[idx_rpts])

            while proposed_flag:
                btstrp_choice = rng_bstrp.choice(org_idx, size=n, replace=True)

                bs_y = respo[btstrp_choice]
                bs_obvrs = obvrs[btstrp_choice]
                bs_unobvrs = unobvrs[btstrp_choice]



                if proposed_flag:
                    # naive method
                    bs_naive_est = minimize(quantile_loss_rdt, ini_prs, args=(bs_obvrs, bs_y, tau, mdl_type),method=optm_method,
                                 tol = tol, bounds = bounds)

                    bs_predi_est = minimize(quantile_loss_rdt, ini_prs, args=(bs_unobvrs, bs_y, tau, mdl_type),method=optm_method,
                                 tol = tol, bounds = bounds)

                    bs_ini_prs = propo_res_navini.x
                    bs_method = root(est_eqs_loop_jstcal_v2, x0 = bs_ini_prs,#naive_est, #ini_prs,
                          method='hybr', tol = tol_hybr, args = (h_kern, bs_y, bs_obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))



                    if bs_method.success == True:
                    #if True:
                        bstrp_propo_res.append( bs_method.x )
                        bstrp_naive_res.append( bs_naive_est.x )
                        bstrp_errfre_res.append( bs_predi_est.x )

                        # recording
                        proposed_succfail[0] += 1

                        if proposed_succfail[0] - proposed_succfail[1] == n_bs:
                            proposed_flag = False

                    else:
                        proposed_succfail[1] += 1

            # -----res
            bstrp_propo_res = np.array(bstrp_propo_res)
            bstrp_naive_res = np.array(bstrp_naive_res)
            bstrp_errfre_res = np.array(bstrp_errfre_res)

            bstrp_std, CI_flag = bs_std_cvg(propo_res_navini.x, prs_truth, bstrp_propo_res, lvl = 0.975)
            grdtru_std, grdtru_CI_flag = bs_std_cvg(predi_res.x, prs_truth, bstrp_errfre_res, lvl = 0.975)
            # naive
            naive_std, naive_CI_flag = bs_std_cvg(naive_res.x, prs_truth, bstrp_naive_res, lvl = 0.975)


        if write_flag:

            handle = open(f'{sim_fd}/hybr.{propo_res_navini.success}-status.{propo_res_navini.status}_simu.{idx_rpts}.xls', 'w')
            handle.write('# x1.range:[%.3f, %.3f]; U.std:%s; eps_prs:[%s, %s]\n' % (unobvrX.T[0].min(), unobvrX.T[0].max(), merr_std_all[0], v_a, v_b))
            handle.write('# x2.range:[%.3f, %.3f]; U.std:%s; eps_prs:[%s, %s]\n' % (unobvrX.T[1].min(), unobvrX.T[1].max(), merr_std_all[1], v_a, v_b))

            handle.write('# Hybr.nfev:: %s;\n' % ( propo_res_navini.nfev))
            handle.write('# Hybr.status:: %s\n' % ( propo_res_navini.status))
            handle.write('# Hybr.message:: %s\n' % ( propo_res_navini.message.replace('\n ', '')))
            handle.write('# bandwidth "h":: %.5f\n' % h_kern)

            handle.write('# True parameters :: %s\n' % (', '.join(prs_truth.astype(str))))
            handle.write('# Oracle ests :: %s\n' % (', '.join(predi_res.x.round(8).astype(str))))
            handle.write('# Proposed ests :: %s\n' % (', '.join(propo_res_navini.x.round(8).astype(str))))
            handle.write('# Naive ests :: %s\n' % (', '.join(naive_res.x.round(7).astype(str))))

            handle.write('# Prop.PredIni ests :: %s\n' % (', '.join(propo_est_prdini)))



            if bstrp_std_indicator and propo_res_navini.success:
                handle.write('# Oracle std :: %s\n' % (', '.join(grdtru_std.round(8).astype(str))))
                handle.write('# Oracle cvg :: %s\n' % (', '.join(grdtru_CI_flag.astype(str))))

                handle.write('# Proposed std :: %s\n' % (', '.join(bstrp_std.round(8).astype(str))))
                handle.write('# Proposed cvg :: %s\n' % (', '.join(CI_flag.astype(str))))

                handle.write('# Naive std :: %s\n' % (', '.join(naive_std.round(8).astype(str))))
                handle.write('# Naive cvg :: %s\n' % (', '.join(naive_CI_flag.astype(str))))



                handle.write('# bstrp :: hybr(%s/%s)\n' % (proposed_succfail[0], proposed_succfail[1]))
                
                #=bstrp__res
                handle.write('# Bootstrap Oracle -> \n')
                bs_prop = [', '.join(each_bs) for each_bs in bstrp_errfre_res.round(5).astype(str)]
                handle.write('\n'.join(bs_prop) + '\n')

                handle.write('# Bootstrap Proposed -> \n' )
                bs_prop = [', '.join(each_bs) for each_bs in bstrp_propo_res.round(5).astype(str)]
                handle.write('\n'.join(bs_prop) + '\n')

                handle.write('# Bootstrap Naive -> \n' )
                bs_prop = [', '.join(each_bs) for each_bs in bstrp_naive_res.round(5).astype(str)]
                handle.write('\n'.join(bs_prop) + '\n')



            handle.close()


