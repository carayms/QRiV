"""
===================
Quantile regression
===================
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = "Arial"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
brightpink = '#FF9EBB'
import scipy.stats as sps
from scipy.optimize import root

from scipy.optimize import minimize
import scipy.special as spsp

from pathlib import Path
sys.path.append('%s/utils' % (str(Path.cwd().parent), ))

from WeiCarrollMethod import weicarroll_qr
from QRfuncV1 import gener_mfunc_v2, quantile_loss_v2, rho_prime_smooth_v2, \
     eps_std, est_eqs_loop_jstcal_v2


import copy
   
plw = 1.5


def bs_std_cvg(orgests, truep, bs_ests, lvl = 0.975):
    bs_std = np.sqrt(np.square(bs_ests - orgests).mean(axis = 0)) 
    bs_half = sps.norm.ppf(lvl, scale = bs_std)

    bs_CIs = np.array([orgests - bs_half, orgests + bs_half])
    olps = bs_CIs - truep
    CI_flag = np.maximum( -np.sign( olps[0] * olps[1]), 0).astype(int)
    return bs_std, CI_flag

if __name__=="__main__":
    # model relathion
    prs_truth = np.array([1, 0.3])
    eps_prs = (1., -0.1)
    simu_num = 1050

    prl_a = int(sys.argv[1])
    prl_b = int(sys.argv[2])
    tau = float(sys.argv[3])
    n = int(sys.argv[4])    
    tol = float(sys.argv[5])
    npsd_data = int(sys.argv[6])
    tuning = float(sys.argv[7])
    
    mdl_type = ['trigonometric', 'square', 'linear'][1]

    n_bs = 200
    perturbation = 0.05
    npsd_bstrp = 99

    form_aprox_step_func = ['algebric', 'logistic', 'normcdf'][-1]
    
    
    h_ord = [-1/2, -1/3, -2/5][1]


    show_flag = [10, False, True][1]
    bstrp_std_indicator = [None, False, True][-1]


    write_flag = [None, False, True][-1]

    # -- - - --
    # Fully Quadrature: Gauss-Hermit normally distributed V.
    intgal_type = ['gauss', 'sampling'][0]
    if intgal_type == 'gauss':
        tdeg = 7
        tGH, wt_tGH = spsp.roots_hermitenorm(tdeg)
        quadratures = Vi, WTi = tGH, wt_tGH/np.sqrt(2*np.pi)
    elif intgal_type == 'sampling':
        rng_sampling = np.random.default_rng(seed=1)
        tdeg = 10000
        tGH = rng_sampling.normal(loc = 0, scale = 1, size = tdeg)
        tGH.sort()
        wt_tGH = 1.0/tdeg
        quadratures = Vi, WTi = tGH, np.ones(tdeg)*wt_tGH
        
    
    output_dir = '%s_tau%s_tol%.e_dtnpsd%s_n%s_bstrp%s.perturb%son2sds.civ2.%snpsd' % (mdl_type, tau, tol, npsd_data, n, n_bs, perturbation, npsd_bstrp)
    
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

    

    # ----
    ntau = 42
    extendtau = np.linspace(0, 1, ntau)

    print('# intgal_type >>', intgal_type, '---vdeg-> %s' % (tdeg))
    if intgal_type == 'gauss':
        print('Vi->', Vi, '\nWi->', WTi, '\nWTi.sum->', WTi.sum())
    else:
        print('# Vi-> \n %s\n %s\n %s' % (Vi[:10], Vi[(tdeg//2-4):(tdeg//2+6)], Vi[-11:]), '\nWi->', WTi[:10], '\nWTi.sum->', WTi.sum())


    start, end = [.5, 2.]

    print('# truth:', prs_truth)
    dsgn_mat_shape = [n, prs_truth.size]

    # sigma_unoise_std
    merr_std = 0.5
    noise_std = 1

    prs_truth = np.array(prs_truth)
    ext_dict = {5000: 1, 1000: 5, 500: 5, 200: 5, 2000: 5, 100:2, 20000: 2}
    supposed_asy_std = {100: np.array([1.2, 0.5]),
                        200: np.array([.3, 0.15]),
                        500: np.array([.3, 0.15]),
                        1000: np.array([.3, 0.15]),
                        2000: np.array([.3, 0.15])}
    ext = ext_dict[n]
    prs_bds = np.array([prs_truth - ext*supposed_asy_std[n],
                        prs_truth + ext*supposed_asy_std[n] ]
                       ).T
    
    print(prs_bds)
    print('\n current tau -> ', tau)
    print('tuning->', tuning)
    #input('check 1')
    

    n_methods = 4
    res_shape = [simu_num, n_methods, prs_truth.size]
    ini_res = np.zeros(res_shape)
    func_optm = np.zeros([simu_num, prs_truth.size])

    hybr_flag = np.zeros(simu_num).astype(bool)
    h_dict = {-1/2: '-1o2', -1/3: '-1o3', -2/5:'-2o5'}
    
    npsd_merr = 1990
    npsd_eps = 1655
    npsd_zvari = 2009
    
    
    rng_data = np.random.default_rng(seed=npsd_data)
    rng_merr = np.random.default_rng(seed=npsd_merr)
    rng_eps = np.random.default_rng(seed=npsd_eps)
    rng_zvari = np.random.default_rng(seed=npsd_zvari)

    rng_inis = np.random.default_rng(seed=0)
    aa, bb = 1-perturbation, 1+perturbation
    pert_noz = rng_inis.uniform(aa, bb, size = (simu_num, prs_truth.size))

    rng_bstrp_seed = np.random.default_rng(seed=npsd_bstrp).choice(np.arange(simu_num*10000), size=simu_num, replace=False)

    

    org_idx = np.arange(n) # this is needed for btsrp
    
    sim_fd = '%s/NmErr_tau%s_n.%s_h.%s.x%s_seed.%s_Quads.%s_errstd.%s_tol.%.e_bstrp.%s' % (output_dir, tau, n, h_dict[h_ord], tuning, npsd_data, tdeg, merr_std, tol, n_bs)
    if not os.path.exists(sim_fd):
        if write_flag:
            try:
                os.mkdir(sim_fd)
            except:
                print('someone has been made it')
    

    
    for i in range(simu_num):
        x = rng_data.uniform(start, end, size=n)
        
        noise_std = eps_prs[0]+eps_prs[1]/np.absolute(x)
        
        eps_nosft = rng_eps.normal(loc = 0, scale = noise_std, size = n)
        eps = eps_nosft - sps.norm.ppf(tau, loc = 0, scale = noise_std)

        u = rng_merr.normal(loc = 0, scale = merr_std, size = n)
        w = x + u

        unobs = np.column_stack([np.ones(n), x])
        obvrs = np.column_stack([np.ones(n), w])

        y = gener_mfunc_v2(prs_truth, unobs, eps, gtype = mdl_type)

        if (i - prl_a) * (i - prl_b) > 0:
            continue
        else:
            pass
            
        if i % 200 == 0:
            print(i, 'processed')

        
        ini_prs = prs_truth * pert_noz[i]
        # 1. loss
        ground_truth = minimize(quantile_loss_v2, ini_prs, args=(unobs, y, tau, mdl_type), method='Nelder-Mead', tol = tol)
        
        # 2. loss on w
        naive_est = minimize(quantile_loss_v2, ini_prs, args=(obvrs, y, tau, mdl_type), method='Nelder-Mead', tol = tol, bounds = prs_bds)
        
        # 3. smooth kernel
        naive_eps = y - gener_mfunc_v2(naive_est.x, obvrs, 0, gtype = mdl_type)
        h_kern = np.power(n, h_ord) * np.square(naive_eps).sum()/(n-prs_truth.size) * tuning

        if i == 0:
            print('n -> %s' % n)
            print('h for proposed method -> %.3f' % h_kern, 'with naive eps rss/(n-n.p) -> %.3f' % (np.square(naive_eps).sum()/(n-prs_truth.size)) )

        smoth_method = root(rho_prime_smooth_v2, ini_prs, args=(unobs, y, tau, h_kern, dsgn_mat_shape, mdl_type, form_aprox_step_func), method='hybr', tol = tol)

        # 4. proposed_method
        proposed_method = root(est_eqs_loop_jstcal_v2, x0 = ini_prs,
                          method='hybr', tol = tol, args = (h_kern, y, obvrs, merr_std, quadratures, tau, prs_bds, mdl_type))



        # est methods over.......^^^^^^^^^
       
    
        hybr_flag[i] = proposed_method.success 
        func_optm[i] = proposed_method.fun

        ini_res[i][0] = ground_truth.x
        ini_res[i][1] = smoth_method.x
        ini_res[i][2] = naive_est.x
        ini_res[i][3] = proposed_method.x


        # bstrp coverage
        hybrid_flag = True; hybrid_succfail = [0, 0]
        weicar_flag = True; weicar_succfail = [0, 0]
        
        bstrp_propose_res = []
        bstrp_naive_res = []
        bstrp_wcr_res = []
        bstrp_grdtru_res = []

        
        if bstrp_std_indicator and proposed_method.success:
            
            rng_bstrp = np.random.default_rng(seed=rng_bstrp_seed[i])
                
            while hybrid_flag or weicar_flag:
                # range(n_bs):
                btstrp_choice = rng_bstrp.choice(org_idx, size=n, replace=True)
                #print(org_idx)

                bs_y = y[btstrp_choice]
                bs_obvrs = obvrs[btstrp_choice]

                bs_unobs = unobs[btstrp_choice]


                if hybrid_flag:
                    bs_naive_est = minimize(quantile_loss_v2, ini_prs, args=(bs_obvrs, bs_y, tau, mdl_type), method='Nelder-Mead', tol = tol)
                    bs_ground_truth = minimize(quantile_loss_v2, ini_prs, args=(bs_unobs, bs_y, tau, mdl_type), method='Nelder-Mead', tol = tol)

                    bs_method = root(est_eqs_loop_jstcal_v2, x0 = ini_prs,
                              method='hybr', tol = tol, args = (h_kern, bs_y, bs_obvrs, merr_std, quadratures, tau, prs_bds, mdl_type))

                    if bs_method.success == True:
                        bstrp_propose_res.append( bs_method.x )
                        bstrp_naive_res.append( bs_naive_est.x )
                        bstrp_grdtru_res.append( bs_ground_truth.x )

                        # recording
                        hybrid_succfail[0] += 1

                        if hybrid_succfail[0] == n_bs:
                            hybrid_flag = False
                    else:
                        hybrid_succfail[1] += 1

        
            # -----res
            bstrp_propose_res = np.array(bstrp_propose_res)
            bstrp_naive_res = np.array(bstrp_naive_res)
            bstrp_grdtru_res = np.array(bstrp_grdtru_res)
            
            if weicarroll_bstrp_indicator:
                bstrp_wcr_res = np.array(bstrp_wcr_res)

            # proposed    
            bstrp_std, CI_flag = bs_std_cvg(proposed_method.x, prs_truth, bstrp_propose_res, lvl = 0.975)

            grdtru_std, grdtru_CI_flag = bs_std_cvg(ground_truth.x, prs_truth, bstrp_grdtru_res, lvl = 0.975)

            # naive
            naive_std, naive_CI_flag = bs_std_cvg(naive_est.x, prs_truth, bstrp_naive_res, lvl = 0.975)


        if write_flag:

            handle = open('%s/hybr.%s-status.%s_simu.%s.xls' % (sim_fd, proposed_method.success, proposed_method.status, i), 'w')
            handle.write('# x.range:[%.3f, %.3f]; U.std:%s; eps_prs:[%s,%s]\n' % (x.min(), x.max(), merr_std, eps_prs[0], eps_prs[1]))
            handle.write('# Hybr.nfev:: %s;\n' % ( proposed_method.nfev))
            handle.write('# Hybr.status:: %s\n' % ( proposed_method.status))
            handle.write('# Hybr.message:: %s\n' % ( proposed_method.message.replace('\n ', '')))
            handle.write('# bandwidth "h":: %.5f\n' % h_kern)
            
            handle.write('# True parameters :: %s\n' % (', '.join(prs_truth.astype(str))))
            handle.write('# Oracle ests :: %s\n' % (', '.join(ground_truth.x.round(8).astype(str))))
            handle.write('# Proposed ests :: %s\n' % (', '.join(proposed_method.x.round(8).astype(str)))) 
            handle.write('# Naive ests :: %s\n' % (', '.join(naive_est.x.round(7).astype(str))))

            if bstrp_std_indicator and proposed_method.success:
                handle.write('# Oracle std :: %s\n' % (', '.join(grdtru_std.round(8).astype(str))))
                handle.write('# Oracle cvg :: %s\n' % (', '.join(grdtru_CI_flag.astype(str))))
                
                handle.write('# Proposed std :: %s\n' % (', '.join(bstrp_std.round(8).astype(str))))
                handle.write('# Proposed cvg :: %s\n' % (', '.join(CI_flag.astype(str))))

                handle.write('# Naive std :: %s\n' % (', '.join(naive_std.round(8).astype(str))))
                handle.write('# Naive cvg :: %s\n' % (', '.join(naive_CI_flag.astype(str))))

                if weicarroll_bstrp_indicator:
                    handle.write('# Wei.Carroll std :: %s\n' % (', '.join(wcr_std.round(8).astype(str))))
                    handle.write('# Wei.Carroll cvg :: %s\n' % (', '.join(wcr_CI_flag.astype(str))))

                handle.write('# bstrp :: hybr(%s/%s), weicar(%s/%s)\n' % (hybrid_succfail[0], hybrid_succfail[1],
                                                                        weicar_succfail[0], weicar_succfail[1]))

            handle.close()

    