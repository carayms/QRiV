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
from sklearn.linear_model import QuantileRegressor
import scipy.stats as sps
from scipy.optimize import root
from scipy.optimize import minimize
import scipy.special as spsp
from pathlib import Path
sys.path.append('%s/utils' % (str(Path.cwd().parent), ))

from WeiCarrollMethod import weicarroll_qr
from QRfuncV1 import gener_mfunc_v2, quantile_loss_v2, rho_prime_smooth_v2, corected_qloss, \
     eps_std, est_eqs_loop_jstcal_v2, bestring
   
plw = 1.5

def bs_std_cvg(orgests, truep, bs_ests, lvl = 0.975):
    bs_std = bs_ests.std(axis = 0, ddof = 1)
    bs_half = sps.norm.ppf(lvl, scale = bs_std)
    bs_CIs = np.array([orgests - bs_half, orgests + bs_half])
    olps = bs_CIs - truep
    CI_flag = np.maximum( -np.sign( olps[0] * olps[1]), 0).astype(int)
    return bs_std, CI_flag

if __name__=="__main__":
    prs_truth = np.array([1, 1])
    v_a, v_b = [1, 0]

    require_n = 1000
    simu_num = int(1.05*require_n)
    
    prl_a = int(sys.argv[1])
    prl_b = int(sys.argv[2])
    tau = float(sys.argv[3])
    n = int(sys.argv[4])    
    tol = float(sys.argv[5])
    npsd_data = int(sys.argv[6])
    tuning = float(sys.argv[7])
    
    n_bs = 100
    perturbation = 0.05
    npsd_bstrp = 99
    mdl_type = ['trigonometric','square', 'linear'][-1]
    form_aprox_step_func = 'normcdf'
    h_ord = [-1/2, -1/3, -2/5][1]

    bstrp_std_indicator = [None, False, True][-1]
    weicarroll_run = [None, False, True][-1]
    write_flag = [None, False, True][-1]

    # -- - - --
    # Fully Quadrature: Gauss-Hermit normally distributed V.
    intgal_type = ['gauss', 'sampling'][0]
    if intgal_type == 'gauss':
        tdeg = 10
        tGH, wt_tGH = spsp.roots_hermitenorm(tdeg)
        quadratures = Vi, WTi = tGH, wt_tGH/np.sqrt(2*np.pi)
        
    elif intgal_type == 'sampling': 
        rng_sampling = np.random.default_rng(seed=1)
        tdeg = 1000
        tGH = rng_sampling.normal(loc = 0, scale = 1, size = tdeg)
        tGH.sort()
        wt_tGH = 1.0/tdeg
        quadratures = Vi, WTi = tGH, np.ones(tdeg)*wt_tGH
        
    output_dir = '%s_tau%s_tol%.e_dtnpsd%s_n%s_bstrp%s.perturb%son2sds.civ1.%snpsd' % (mdl_type, tau, tol, npsd_data, n, n_bs, perturbation, npsd_bstrp)
    
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

    print('# truth:', prs_truth)
    dsgn_mat_shape = [n, prs_truth.size]

    # sigma_u
    merr_std = 0.5
    noise_std = 0.5

    
    ext_dict = {5000: 3, 1000: 3, 500: 3, 200: 3, 100:3, 2000:3, 20000:3}
    supposed_asy_std = {100: np.array([.5, .1]),
                        200: np.array([2.5, 0.5]),
                        500: np.array([1.5, 0.2]),
                        1000: np.array([1.5, 0.2]),
                        2000: np.array([.3, 0.1]),
                        5000: np.array([1.5, 0.2]),
                        20000: np.array([.3, 0.1])}
    ext = ext_dict[n]
    prs_bds = np.array([prs_truth - ext*supposed_asy_std[n],
                        prs_truth + ext*supposed_asy_std[n] ]
                       ).T
    wcl_bds = np.array([[-1.8, 3.8], [0.5, 1.5]])
    print(prs_bds)
    print('\n current tau -> ', tau)
    print('tuning->', tuning)
    #input('check 1')
    

    n_methods = 3
    res_shape = [simu_num, n_methods, prs_truth.size]
    ini_res = np.zeros(res_shape)

    func_optm = np.zeros([simu_num, prs_truth.size])

    hybr_flag = np.zeros(simu_num).astype(bool)
    h_dict = {-1/2: '-1o2', -1/3: '-1o3', -2/5:'-2o5'}
    #input('check1')
    
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
    
    sim_fd = '%s/tau%s_n.%s_h.%s.x%s_seed.%s_Quads.%s_errstd.%s_tol.%.e_bstrp.%s' % (output_dir, tau, n, h_dict[h_ord], tuning, npsd_data, tdeg, merr_std, tol, n_bs)
    if not os.path.exists(sim_fd):
        if write_flag:
            try:
                os.mkdir(sim_fd)
            except:
                print('someone has been made it')

    
    for i in range(simu_num):
        x = rng_data.uniform(5, 5+12**0.5, size=n)

        noise_std = v_a + v_b*x
        eps_nosft = rng_eps.normal(loc = 0, scale = noise_std, size = n)
        epsi = eps_nosft - sps.norm.ppf(tau, loc = 0, scale = noise_std)

        u = rng_merr.normal(loc = 0, scale = merr_std, size = n)
        w = x + u

        unobs = np.column_stack([np.ones(n), x])
        obvrs = np.column_stack([np.ones(n), w])

        y = gener_mfunc_v2(prs_truth, unobs, epsi, gtype = mdl_type)

        if (i - prl_a) * (i - prl_b) > 0:
            #print('no')
            continue
        else:
            pass
            #print('%s-"th" starts' % (i), end = ' | ')
        if i % 200 == 0:
            print(i, 'processed')

        
        ini_prs = prs_truth * pert_noz[i]
        # 1. ground truth: loss
        ground_truth = minimize(quantile_loss_v2, ini_prs, args=(unobs, y, tau), method='Nelder-Mead', tol = tol)
        
        # 2. naive: loss on w
        naive_est = minimize(quantile_loss_v2, ini_prs, args=(obvrs, y, tau), method='Nelder-Mead', tol = tol)#
        
        # the method: smooth kernel on x
        naive_eps = y - gener_mfunc_v2(naive_est.x, obvrs, 0, gtype = mdl_type)
        h_base = np.power(n, h_ord) * np.std(naive_eps, ddof = prs_truth.size)
        h_kern = h_base * tuning

        # !3. weicar's method
        if weicarroll_run:
            beta_weicar = weicarroll_qr(tau, extendtau, obvrs, y, merr_std, wcl_bds)

        
        # 3. wsz method
        wsz_res = minimize(corected_qloss, ini_prs, args=(y, obvrs, merr_std, tau, h_kern, 'meanLoss'), method='L-BFGS-B', bounds = prs_bds, tol = 1e-15)#'L-BFGS-B'
        
        # 5. proposed_method
        
        if i == 0:
            print('n -> %s' % n)
            print('h for proposed method -> %.3f' % h_kern, 'with naive eps std -> %.3f' % np.std(naive_eps, ddof = 1) )

        proposed_method = root(est_eqs_loop_jstcal_v2, x0 = ini_prs, 
                          method='hybr', tol = tol, args = (h_kern, y, obvrs, merr_std, quadratures, tau, prs_bds))


        # est methods over.......^^^^^^^^^

        subtc = np.sign(proposed_method.x - prs_bds.T)
        if sum(subtc[0] * subtc[1]) != -prs_truth.size:
            proposed_method.success = False
       
        #input(proposed_method)
        hybr_flag[i] = proposed_method.success
        func_optm[i] = proposed_method.fun

        ini_res[i][0] = naive_est.x
        ini_res[i][1] = proposed_method.x
        ini_res[i][2] = beta_weicar


        # bstrp coverage
        proposed_flag = True; proposed_succfail = [0, 0]
        weicar_flag = True; weicar_succfail = [0, 0]
        naive_flag = True; naive_succfail = [0, 0]
        cle_flag = True; cle_succfail = [0, 0]
        
        bstrp_propose_res = []
        bstrp_naive_res = []
        bstrp_wcr_res = []
        bstrp_cle_res = []

        
        if bstrp_std_indicator and proposed_method.success:
            rng_bstrp = np.random.default_rng(seed=rng_bstrp_seed[i])
            while proposed_flag or weicar_flag:
                btstrp_choice = rng_bstrp.choice(org_idx, size=n, replace=True)
                #print(org_idx)

                bs_y = y[btstrp_choice]
                bs_obvrs = obvrs[btstrp_choice]

                if proposed_flag:
                    # naive method
                    bs_naive_est = minimize(quantile_loss_v2, ini_prs, args=(bs_obvrs, bs_y, tau), method='Nelder-Mead', tol = tol)#, bounds = prs_bds)

                    bs_method = root(est_eqs_loop_jstcal_v2, x0 = ini_prs, #proposed_method.x, #x0 = ini_prs,est_eqs_loop_jstcal_v2
                              method='hybr', tol = tol, args = (h_kern, bs_y, bs_obvrs, merr_std, quadratures, tau, prs_bds))
                    bs_wsz_res = minimize(corected_qloss, ini_prs, args=(bs_y, bs_obvrs, merr_std, tau, h_kern, 'meanLoss'), method='L-BFGS-B', bounds = prs_bds, tol = 1e-15)

                    if bs_method.success == True:
                        bstrp_propose_res.append( bs_method.x )
                        bstrp_naive_res.append( bs_naive_est.x )

                        # recording
                        proposed_succfail[0] += 1
                        naive_succfail[0] += 1
                        weicar_succfail[0] += 1

                        if proposed_succfail[0] == n_bs:
                            proposed_flag = False
                        if naive_succfail[0] == n_bs:
                            naive_flag = False

                        if weicar_succfail[0] < n_bs:
                            bs_weicar = weicarroll_qr(tau, extendtau, bs_obvrs, bs_y, merr_std, wcl_bds)
                            bstrp_wcr_res.append( bs_weicar )
                        if cle_succfail[0] < n_bs:
                            bstrp_cle_res.append( bs_wsz_res.x )
                    else:
                        proposed_succfail[1] += 1
                        naive_succfail[1] += 1
                        weicar_succfail[1] += 1
                        cle_succfail[1] += 1


            # -----res
            bstrp_propose_res = np.array(bstrp_propose_res)
            bstrp_naive_res = np.array(bstrp_naive_res)
            bstrp_cle_res = np.array( bstrp_cle_res )
            bstrp_wcr_res = np.array(bstrp_wcr_res)

            # proposed    
            bstrp_std, CI_flag = bs_std_cvg(proposed_method.x, prs_truth, bstrp_propose_res, lvl = 0.975)

            # naive
            naive_std, naive_CI_flag = bs_std_cvg(naive_est.x, prs_truth, bstrp_naive_res, lvl = 0.975)

            # corrected-loss estimator
            cle_std, cle_CI_flag = bs_std_cvg(wsz_res.x, prs_truth, bstrp_cle_res, lvl = 0.975)

            # wei-carroll
            wcr_std, wcr_CI_flag = bs_std_cvg(beta_weicar, prs_truth, bstrp_wcr_res, lvl = 0.975)

        
        if write_flag:

            handle = open('%s/hybr.%s-status.%s_simu.%s.xls' % (sim_fd, proposed_method.success, proposed_method.status, i), 'w')
            handle.write('# x.range:[%.3f, %.3f]; U.std:%s; mdl.std:%s+%s*x\n' % (x.min(), x.max(), merr_std, v_a, v_b))
            handle.write('# Hybr.nfev:: %s;\n' % ( proposed_method.nfev))
            handle.write('# Hybr.status:: %s\n' % ( proposed_method.status))
            handle.write('# Hybr.message:: %s\n' % ( proposed_method.message.replace('\n ', '')))
            handle.write('# bandwidth "h":: %.5f\n' % h_kern)
            
            handle.write('# True parameters :: %s\n' % (', '.join(prs_truth.astype(str))))
            handle.write('# ests Proposed :: %s\n' % (', '.join(proposed_method.x.round(8).astype(str)))) 
            handle.write('# ests Naive :: %s\n' % (', '.join(naive_est.x.round(7).astype(str))))
            handle.write('# ests WeiCarroll :: %s\n' % (', '.join(beta_weicar.round(7).astype(str))))
            handle.write('# CLE ests :: %s\n' % (', '.join(wsz_res.x.round(7).astype(str))))


            if bstrp_std_indicator and proposed_method.success:
                handle.write('# std Proposed :: %s\n' % (', '.join(bstrp_std.round(8).astype(str))))
                handle.write('# std Naive :: %s\n' % (', '.join(naive_std.round(8).astype(str))))
                handle.write('# std WeiCarroll :: %s\n' % (', '.join(wcr_std.round(8).astype(str))))
                handle.write('# std CLE  :: %s\n' % (', '.join(cle_std.round(8).astype(str))))

                handle.write('# cvg Proposed :: %s\n' % (', '.join(CI_flag.astype(str))))
                handle.write('# cvg Naive :: %s\n' % (', '.join(naive_CI_flag.astype(str))))
                handle.write('# cvg WeiCarroll :: %s\n' % (', '.join(wcr_CI_flag.astype(str))))
                handle.write('# cvg CLE  :: %s\n' % (', '.join(cle_CI_flag.astype(str))))

                
                


                handle.write('# bstrp :: hybr(%s/%s), naive_bstrp(%s/%s), weicarroll_brstry(%s/%s), cle_brstry(%s/%s)\n' % (proposed_succfail[0], proposed_succfail[1],
                                                                                                naive_succfail[0], naive_succfail[1],
                                                                                                weicar_succfail[0], weicar_succfail[1],
                                                                                                cle_succfail[0], cle_succfail[1]))
     
                
                handle.write('# BSouts Proposed  :: %s\n' % (', '.join( bestring(bstrp_propose_res) )))
                handle.write('# BSouts Naive  :: %s\n' % (', '.join(bestring(bstrp_naive_res) )))
                handle.write('# BSouts WeiCaroll  :: %s\n' % (', '.join(bestring(bstrp_wcr_res) )))
                handle.write('# BSouts CLE  :: %s\n' % (', '.join(bestring(bstrp_cle_res) )))

            handle.close()

       

 
