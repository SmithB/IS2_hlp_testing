#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:21:46 2025

@author: ben
"""

import numpy as np
import pointCollection as pc
import matplotlib.pyplot as plt
import ATL11
import os

# Reorganize ATL06 data into pairs
def to_pairs(D6):
    _, crp_dict = pc.unique_by_rows(np.c_[D6.cycle, D6.rgt, D6.pair], return_dict=True)

    pair_data=[]
    for crp, ii in crp_dict.items():
        D6sub = D6[ii]
        useg = np.unique(D6sub.segment_id)
        D_pair = pc.data().from_dict({field:np.zeros((len(useg), 2))+np.nan for field in D6.fields})
        for beam in [0., 1.]:
            D6sub1 = D6sub[D6sub.beam==beam]
            ii =np.searchsorted(useg, D6sub1.segment_id)
            for field in D6.fields:
                getattr(D_pair, field)[ii, int(beam)]=getattr(D6sub1, field)
        pair_data += [D_pair]
    return pair_data

def to_pc(D11):
    D=D11.ROOT
    N_pts = len(D.latitude)
    N_cycles = len(D.cycle_number)
    cycles = np.array(D.cycle_number)
    D_out = pc.data(columns=N_cycles).from_dict({'pair_num':np.zeros((N_pts, N_cycles))+D11.pair_num})
    for field in set(D.list_of_fields):

        if field=='cycle_number':
            D_out.assign(cycle=np.tile(cycles[None,:], [N_pts,1]))
            continue
        if field.ndim==2:
            D_out.assign({field:getattr(D, field)})
            continue
        if field.shape == (N_pts,):
            D_out.assign({field:np.title(field[:, None], [1, N_cycles])})
    D_out.__update_size_and_shape__()
    return D_out

def assign_ATL06_fields(D6, sigma_geo_xy=3.5, sigma_geo_r = 0.03):
    D6.assign(dh_fit_dy = np.tile(((D6.z[:,1]-D6.z[:,0])/(D6.y_atc[:,1]-D6.y_atc[:,0]))[:, None], [1, 2]))
    D6.assign(dh_fit_dy_sigma = D6.h_li_sigma*np.sqrt(2)/90)
    slope_mag = np.sqrt(D6.dh_fit_dx**2+D6.dh_fit_dy**2)

    D6.assign(dh_geoloc = slope_mag*np.sqrt(D6.sigma_geo_xt**2+D6.sigma_geo_at**2))
    D6.assign(h_li=D6.z,
              delta_time = D6.t *24*3600*365.25,
             h_li_sigma = 0.6/np.sqrt(57)+np.zeros_like(D6.x),
              h_rms_misfit = 0.16 +np.ones_like(D6.x),
             cycle_number=D6.cycle.astype(int),
             BP=D6.pair,
             atl06_quality_summary=np.zeros_like(D6.x),
             sigma_geo_xt = np.zeros_like(D6.x)+sigma_geo_xy,
             sigma_geo_at = np.zeros_like(D6.x) + sigma_geo_xy,
             sigma_geo_r = np.zeros_like(D6.x) + sigma_geo_r,
            sigma_geo_h = D6.dh_geoloc.copy(),
             r_eff = np.ones_like(D6.x),
             )
    D6.assign({field:np.zeros_like(D6.x) for field in ['bsnow_h', 'bsnow_conf',
                                                   'cloud_flg_asr','cloud_flg_atm',
                                                   'tide_ocean','dac','snr_significance',
                                                  'signal_selection_source','geoid_h',
                                                  'geoid_free2mean']})

def make_sim_ATL11_data(data, cycles=None, D11_root=None, subregion=0):

    if cycles is None:
        cycles = [np.min(data.cycle.astype(int)), np.max(data.cycle.astype(int))]

    _, rp_dict = pc.unique_by_rows(np.c_[data.rgt, data.pair], return_dict=True)

    D11_list=[]
    for rp, ii in rp_dict.items():
        D6 = data[ii]

        ref_pt_numbers = np.unique(np.round(D6.segment_id/3)*3)
        D6 = pc.data(columns=2).from_list(to_pairs(D6))
        assign_ATL06_fields(D6)
        _, ref_pt_numbers, ref_pt_x = ATL11.select_ATL06_data(D6)
        D11_pts = ATL11.data().from_ATL06(D6, beam_pair=rp[1], cycles=cycles,
                                      ref_pt_numbers=ref_pt_numbers, ref_pt_x=ref_pt_x, hemisphere=1)

        D11=ATL11.data.from_list(D11_pts)
        D11.track_num, D11.beam_pair = rp
        setattr(D11.cycle_stats,'cycle_number',list(range(cycles[0],cycles[1]+1)))
        setattr(D11.ROOT,'cycle_number',list(range(cycles[0],cycles[1]+1)))

        if D11_root is not None:
            if not os.path.isfile(D11_root):
                os.mkdir(D11_root)
            D11_filename = f'ATL11_{rp[0]:%02d}{subregion:%02d}_{cycles[0]:%02d}{cycles[1]:%02d}_999_99.h5'
            D11.write_to_file(os.path.join(D11_root, D11_filename))
        else:
            D11_list += [to_pc(D11)]

    if D11_root is None:
        return D11_list
    else:
        return None


# test:
if False:
    data=pc.data().from_h5('/Users/ben/temp/sim_ALT06_data_with_errors.h5')
    
    rp, rp_dict = pc.unique_by_rows(np.c_[data.rgt, data.pair], return_dict=True)
    data=pc.data.from_list([
        data[rp_dict[jj]] for jj in rp[0:3]
        ])
    D11 = make_sim_ATL11_data(data)


