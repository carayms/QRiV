import os.path
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl

from pandas.core.common import random_state


mpl.use('TkAgg')
# === Set global font ===

# Force DejaVu Sans (it supports basic Unicode flowers)
mpl.rcParams['font.family'] = ['DejaVu Sans', 'Arial'][1]
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
# import counter class from collections module
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, expit



from scipy.optimize import minimize, root
import scipy.special as spsp
import scipy.stats as sps

from numba import jit

#from numba_stats import norm

import realDataRuncs as Funcs
import importlib
importlib.reload(Funcs)
import pickle

#from QRfuncV1 import quantile_loss_v2

def quantile_loss_rdt(params, x, y, tau, gtype = 'none'):
    if gtype == 'none':
        raise ValueError(f"[gtype] must be specified, not {gtype}")

    residuals = y - rdt_mfunc(params, x, gtype)
    #posflg = residuals >= 0
    #print('%s out of %s are positive...' % (sum(posflg), y.size))
    rtn = np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))
    return rtn


def rdt_mfunc(prs, cvats, gtype = 'cherry'):
    #print(prs.shape, cvats.shape)
    # 1, temp_mean  meter  precipitation        lat         lon
    #
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
        #print(phi_in_scl.shape)

        phi_out = sps.norm.cdf(phi_in_scl, loc = 0, scale = 1) + qtl - 1
        phi_out += phi_in * sps.norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker

        #phi_out = norm.cdf(phi_in_scl, 0.0, 1.0) + qtl - 1
        #phi_out += phi_in * norm.pdf(phi_in_scl, loc = 0, scale = 1)/h_ker

        eq1s = phi_out
        eq2s = phi_out * np.expand_dims(extrop_X1, axis = 1)
        eq3s = phi_out * np.expand_dims(varib_X2, axis = 0)
        eq4s = phi_out * O_i[2]
        eq5s = phi_out * O_i[3]
        eq6s = phi_out * O_i[4]

        #eq_forming = np.array([eq1s, eq2s, eq3s, eq4s]) * wts
        eq_forming = np.array([eq1s, eq2s, eq3s, eq4s, eq5s, eq6s]) * wts
        #q_forming = np.array([eq1s, eq2s, eq3s, eq4s, eq5s]) * wts

        outer_eqs[idx_o] = np.sum(eq_forming.real, axis = (1, 2))



    rtns = np.mean(outer_eqs, axis = 0)

    #outer_eqs_in = np.zeros([len(Ws), prs.size])

    #input(rtns)
    #......

    return rtns

if __name__=="__main__":
    n_repts = 1050
    tau = 0.5 
    tol = 1e-7
    tuning = 1
    idx_rpts = int(sys.argv[1])


    mdl_type = ['trigonometric', 'square', 'linear', 'cherry'][-1]
    covts = ['temp_mean', 'meter', 'precipitation', 'lat', 'lon']
    predi_covts = ['temp_mean_predi', 'meter_predi', 'precipitation', 'lat', 'lon']

    n = 903
    n_prs = len(predi_covts)+1
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


    # === Load datasets ===
    main_data_path = f"J.cherry.blossoms_2024-02.29-03.18_predi-meter.tempmean-_by.14-18.days.csv"
    df_ini = pd.read_csv(main_data_path)
    df_ini['date'] = pd.to_datetime(df_ini['date'])

    #------
    ctring_varibs = ['lat', 'lon', 'precipitation', 'temp_mean', 'temp_mean_predi']
    ctring_flag = True
    if ctring_flag:
        df_ini[ctring_varibs]  = df_ini[ctring_varibs] - df_ini[ctring_varibs].mean(axis = 0)

    # algo set up
    optm_method = ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'TNC'][0]
    bounds = np.tile([-100, 100], n_prs).reshape(n_prs, 2)


    #--------select one date
    grps = df_ini['code']

    npsd_data = 1652
    rng = np.random.default_rng(npsd_data)
    rng_seed = rng.choice(np.arange(999999), size=n_repts, replace=False)

    ini_seeds = np.random.default_rng(seed=0).choice(np.arange(1000000), size=n_repts, replace=False)
    # ---- bootstrap and inin
    npsd_bstrp = 99
    org_idx = np.arange(n) # this is needed for btsrp
    rng_bstrp_seed = np.random.default_rng(seed=npsd_bstrp).choice(1000000, size=n_repts, replace=False)
    # ---- bootstrap and inin over

    est_shape = [n_repts, n_prs]
    est_arsenal = {'predi_ests': np.zeros(est_shape),
                   'naive_ests': np.zeros(est_shape),
                   'propo_est_navini': np.zeros(est_shape)}



    rng_one = np.random.default_rng(rng_seed[idx_rpts])
    ini_prs = np.random.default_rng(seed=ini_seeds[idx_rpts]).normal(0, 1, n_prs)

    df = Funcs.sample_one_per_group(df_ini, grps, rng_one)   
    unobvrs = df[predi_covts].to_numpy()

    # 1. the response variavle
    respo = df['nowToKaika_days'].to_numpy()


    # 2.  observed
    obvrs = df[covts].to_numpy()


    predi_res = minimize(quantile_loss_rdt, ini_prs, args=(unobvrs, respo, tau, mdl_type),method=optm_method,#'Nelder-Mead',
                             tol = tol, bounds = bounds)
    predi_est = ['%.4f' % i for i in predi_res.x]

    naive_res = minimize(quantile_loss_rdt, ini_prs, args=(obvrs, respo, tau, mdl_type),method=optm_method,
                             tol = tol, bounds = bounds)
    naive_est = ['%.4f' % i for i in naive_res.x]


     # smooth
    naive_eps = respo - gener_mfunc_v2(naive_res.x, obvrs, 0, gtype = mdl_type)
    h_base = np.power(n, h_ord) * np.std(naive_eps, ddof = n_prs)
    h_kern = h_base * tuning

    propo_res_navini = root(est_eqs_loop_jstcal_v2, x0 = naive_res.x,#propo_ini*pert_noz[-1],#predi_est.x,#naive_est, #ini_prs,
                      method='hybr', tol = 1e-7, args = (h_kern, respo, obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))


    propo_est_navini = ['%.4f' % i for i in propo_res_navini.x]


    print(f'-----> propo by navini status:{propo_res_navini.status}, success:{propo_res_navini.success}, message:{ propo_res_navini.message}')
    print(f'\n---> fun.loss:{propo_res_navini.fun.round(2)}')


    print()

    print(f"tau->{tau}")
    print(f"naive_est\t{'\t'.join(naive_est)}")
    print(f"errfe_est\t{'\t'.join(predi_est)}")
    print(f"propo_est\t{'\t'.join(propo_est_navini)} by navini")


    # bstrp coverage
    proposed_flag = True; proposed_succfail = [0, 0]

    bstrp_propo_res = []
    bstrp_naive_res = []
    bstrp_errfre_res = []



    if bstrp_std_indicator:

        rng_bstrp = np.random.default_rng(seed=rng_bstrp_seed[idx_rpts])

        while proposed_flag:

            # range(n_bs):
            btstrp_choice = rng_bstrp.choice(org_idx, size=n, replace=True)
            #print(org_idx)

            bs_y = respo[btstrp_choice]
            bs_obvrs = obvrs[btstrp_choice]
            bs_unobvrs = unobvrs[btstrp_choice]



            if proposed_flag:
                # naive method
                bs_naive_est = minimize(quantile_loss_rdt, ini_prs, args=(bs_obvrs, bs_y, tau, mdl_type),method=optm_method,
                             tol = tol, bounds = bounds)

                bs_predi_est = minimize(quantile_loss_rdt, bs_naive_est.x, args=(bs_unobvrs, bs_y, tau, mdl_type),method=optm_method,
                             tol = tol, bounds = bounds)
                
                bs_method = root(est_eqs_loop_jstcal_v2, x0 = bs_naive_est.x,#naive_est, #ini_prs,
                      method='hybr', tol = 1e-7, args = (h_kern, bs_y, bs_obvrs, merr_std_all, quadratures, tau, bounds, mdl_type))

                if bs_method.success == True:
                    bstrp_propo_res.append( bs_method.x )
                    bstrp_naive_res.append( bs_naive_est.x )
                    bstrp_errfre_res.append( bs_predi_est.x )

                    # recording
                    proposed_succfail[0] += 1

                    if proposed_succfail[0] % 50 == 0:
                        print(f'{proposed_succfail[0]} bootstraps processed..')

                    if proposed_succfail[0] == n_bs:
                        proposed_flag = False
                else:
                    proposed_succfail[1] += 1

                

        print('proposed_succfail ->', proposed_succfail)

        
        # -----res
        bstrp_propo_res = np.array(bstrp_propo_res)
        bstrp_naive_res = np.array(bstrp_naive_res)
        bstrp_errfre_res = np.array(bstrp_errfre_res)

        print(f'bstrp_propo_res- > {bstrp_propo_res.shape}')
        print(f'bstrp_naive_res- > {bstrp_naive_res.shape}')
        print(f'bstrp_errfre_res- > {bstrp_errfre_res.shape}')

        z_score_naive = naive_res.x/bstrp_naive_res.std(ddof = 1, axis = 0)
        z_score_errfre = predi_res.x/bstrp_errfre_res.std(ddof = 1, axis = 0)
        z_score_propo = propo_res_navini.x/bstrp_propo_res.std(ddof = 1, axis = 0)

        print(f'naive_std.hat\t{'\t'.join(bstrp_naive_res.std(ddof = 1, axis = 0).round(4).astype(str))}')
        print(f'errfe_std.hat\t{'\t'.join(bstrp_errfre_res.std(ddof = 1, axis = 0).round(4).astype(str))}')
        print(f'propo_std.hat\t{'\t'.join(bstrp_propo_res.std(ddof = 1, axis = 0).round(4).astype(str))}')
        #print()


        print(f'naive Z\t{'\t'.join(z_score_naive.round(3).astype(str))}\nerrFe Z\t{'\t'.join(z_score_errfre.round(3).astype(str))}')
        print(f'propo Z\t{'\t'.join(z_score_propo.round(3).astype(str))}')
        #print()

        naive_pval = 2*(1 - sps.norm.cdf(np.absolute(z_score_naive)))
        errfre_pval = 2*(1 - sps.norm.cdf(np.absolute(z_score_errfre)))
        propo_pval = 2*(1 - sps.norm.cdf(np.absolute(z_score_propo)))

        print(f'naive P\t{'\t'.join(naive_pval.round(5).astype(str))}\nerrFr P\t{'\t'.join(errfre_pval.astype(str))}')
        print(f'propo P\t{'\t'.join(propo_pval.round(5).astype(str))}')


        #------




    ci_level = [[2.5, 97.5], [5, 95]][0]
    upper_ci = ci_level[1]/100

    naive_mean = naive_res.x
    predi_mean = predi_res.x
    propo_mean = propo_res_navini.x

    erba_sft = 0.2
    std_naive = bstrp_naive_res.std(ddof = 1, axis = 0)
    std_predi = bstrp_errfre_res.std(ddof = 1, axis = 0)
    std_propo = bstrp_propo_res.std(ddof = 1, axis = 0)

    dump_file = f"Fig_centering.{ctring_flag}_realdat{tau}_by.rdmOne.pkl"
    data_all = [[naive_res, predi_res, propo_res_navini],
                [bstrp_naive_res,bstrp_errfre_res,  bstrp_propo_res]]
    with open(dump_file, 'wb') as f:
        pickle.dump(data_all, f)

    fig = plt.figure(figsize=(8, 4.2))
    nrows = 2
    ncols = np.ceil(n_prs/nrows).astype(int)
    gs = gridspec.GridSpec(nrows = nrows, ncols=ncols, wspace = .35, hspace = .31, height_ratios=[1, 1], width_ratios = [1, 1, 1],
                           left=0.07, right=0.77, top= .96, bottom = 0.1)
    axes = []


    cpsz = 4
    mksz = 5
    lw = 1.5
    colors = ['#5891bd', '#e3a144', '#ed8293']
    plot_keys = ['Naive', 'Interpolation', "Proposed"]
    varib_names = ['intercept', 'daily avg. temperature', 'meter', 'precipitation', 'latitude', 'longitude']
    step_of_yticks = [0.1, 0.2, 0.2, 0.01, 0.2, 0.05]
    starts_of_yticks = [1.4, -0.8, -3.6, -0.04, 0.4, -0.15]
    end_of_yticks = [1.78, -0.1, -2.9, 0.01, 1.1, 0.04]
    for i in range(n_prs):
        a, b = divmod(i, ncols)
        ax = fig.add_subplot(gs[a, b])
        axes.append(ax)
        ax.errorbar(-erba_sft,naive_mean[i], yerr = sps.norm.ppf(upper_ci) * std_naive[i], elinewidth = lw,mew = lw,
                    marker = 'D', capsize = cpsz,capthick=lw, ecolor = colors[0], mfc = 'white', mec = colors[0], ms = mksz)
        ax.errorbar(0, predi_mean[i], yerr = sps.norm.ppf(upper_ci) * std_predi[i], elinewidth = lw,mew = lw,
                        marker = 'D', capsize = cpsz,capthick=lw, ecolor = colors[1], mfc = 'white', mec = colors[1], ms = mksz)
        ax.errorbar(erba_sft, propo_mean[i], yerr = sps.norm.ppf(upper_ci) * std_propo[i], elinewidth = lw, mew = lw,
                        marker = 'D', capsize = cpsz,capthick=lw, ecolor = colors[2], mfc = 'white', mec = colors[2], ms = mksz)

        ax.set_xticks([])
        ax.set_xlabel(r'$\beta_{%s}$ (%s)' % (i, varib_names[i]), fontsize = 12)


        yticks = np.arange(starts_of_yticks[i], end_of_yticks[i], step_of_yticks[i]).round(2)
        ax.set_yticks(yticks)

    [ax.set_xlim([-0.45, 0.45]) for ax in axes]

    from matplotlib.lines import Line2D
    legend_ax = fig.add_axes([0.82, 0.12, 0.15, 0.8])
    # Turn off the axes frame
    legend_ax.axis('off')

    # Create custom legend elements
    legend_elements = [Line2D([0], [0], color=i, lw=2, label=j) for i, j in zip(colors, plot_keys)]
    # Add the legend to the independent axes
    legend_ax.legend(handles=legend_elements, loc='center', fontsize = 12)


    plt.savefig(f'Fig_centering.{ctring_flag}_realdat{tau}_by.rdmOne.pdf')



    catgs = ['est', r'$\wh{\rm std}$', r'$p$-value']
    table = [[naive_res.x, predi_res.x, propo_res_navini.x],
             [std_naive, std_predi, std_propo],
             [naive_pval, errfre_pval,  propo_pval]]


    for idx_cat, cat_name in enumerate(catgs):
        print(rf" {cat_name} &  &  &  &  &  &  \\")
        for idx_method, each in enumerate(table[idx_cat]):
            method = plot_keys[idx_method]

            if idx_cat < 2:
                line = rf"{method} & {' & '.join(np.char.mod('%.4f', each))} \\"
            else:
                reform_line = ['%.1e' % i if i < 1e-4 else '%.4f' % i for i in each ]
                reform_line = [ec if float(ec) != 0.0 else '0.0' for ec in reform_line]
                line = rf"{method} & {' & '.join(reform_line)} \\"
            print(line)

    plt.show()

