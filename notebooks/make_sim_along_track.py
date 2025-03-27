#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:03:15 2025

@author: ben
"""
import numpy as np
import pointCollection as pc
import h5py


def R_WGS84(latitude):
    # return the local radius of the WGS84 ellipsoid

    Re = 6378137.0
    #flattening
    f = 1/298.25722356
    return Re * (1-f*np.sin(latitude*np.pi/180)**2)

def calc_xatc(rgt1):

    # calculate the along-track distance measured along rgt1

    dLat = np.diff(rgt1.latitude)
    dLon = np.diff(np.unwrap(rgt1.longitude, period=360))

    r = R_WGS84(rgt1.latitude)
    scale = (np.cos(rgt1.latitude[1:]*np.pi/180) + np.cos(rgt1.latitude[0:-1]*np.pi/180))*0.5
    dS = (r[0:-1]+r[1:])/2* np.pi/180* np.sqrt(dLat**2 + (dLon*scale)**2)
    dist = np.append(0, np.cumsum(dS))
    return dist

def shift_atc(D, dxy, ind=None, inplace=True, assign_vectors=True, return_new=False):
    '''
    Shift a track by dxy in along-track coordinates
    '''

    if ind is None:
        ind=np.arange(D.size, dtype=int)

    if 'v_atc_x' not in D.fields:
        s_hat = np.diff(D.x+1j*D.y)
        s_hat /= np.abs(s_hat)
        s_hat=np.append(s_hat, s_hat[-1])
        if assign_vectors:
            D.assign(v_atc_x = np.real(s_hat))
            D.assign(v_atc_y = np.imag(s_hat))
    else:
        s_hat = D.v_atc_x +1j*D.v_atc_y
    delta = s_hat[ind] * dxy[0] + 1j*s_hat[ind]*dxy[1]

    if inplace:
        # return_new is false, shift the original structure
        D.x[ind] += np.real(delta)
        D.y[ind] += np.imag(delta)
        return
    # return_new is true, make a new structure with 'x' and 'y' fields
    x1 = D.x
    y1 = D.y
    x1[ind] += np.real(delta)
    y1[ind] += np.imag(delta)

    if return_new:
        return pc.data().from_dict({'x':x1,'y':y1})
    else:
        return x1, y1

def to_xy(latitude, longitude, lat_0, lon_0):

    # convert latitude and longitudue to platte-carre x, y

    r=R_WGS84(lat_0)
    lat_scale=(r*2*np.pi/360.)
    lon_scale=(r*2*np.pi/360.*np.cos(lat_0*np.pi/180))
    x=(longitude-lon_0)*lon_scale
    y=(latitude-lat_0)*lat_scale
    return x, y

def to_latlon(x, y, lat_0, lon_0):

    # convert platte-carre x, y to latitude, longitude

    r=R_WGS84(lat_0)
    lat_scale=(r*2*np.pi/360.)
    lon_scale=(r*2*np.pi/360.*np.cos(lat_0*np.pi/180))
    longitude = lon_0 + (x/lon_scale)
    latitude = lat_0 + (y/lat_scale)
    return latitude, longitude

def calc_azimuth(D, lat0, lon0):

    # calculate the azimuth of a track WRT local north
    if D.EPSG is None:
        x1, y1 = to_xy(D.latitude+0.001, D.longitude, lat0, lon0)
    else:
        temp=pc.data().from_dict({'latitude':D.latitude.copy(),
                                  'longitude':D.longitude.copy()})
        temp.latitude += 0.001
        temp.get_xy(EPSG=D.EPSG)
        x1, y1 = [temp.x, temp.y]

    n_hat = (x1+1j*y1) - (D.x+1j*D.y)
    n_hat /= np.abs(n_hat)
    track_hat = D.v_atc_x + 1j*D.v_atc_y
    D.assign(seg_azimuth = np.angle(track_hat * np.conj(n_hat)))

def apply_shifts(D, key_vars, sigma, dims=['y'], shifts=None, shift_dict=None, update_yatc=False):
    '''
    Shift the tracks in D in along-track coordinates.


    Parameters
    ----------
    D : pointColection.data
        data structures.
    key_vars : list
        fields in D.  One shift will be applied for each unique combination of these
    sigma : float
        standard deviation of shifts.
    dims : list, optional
        directions in which tracks will be shifted. The default is ['y'].
    shifts : numpy array, optional
        Optional array of shifts  The array must have one column for each
        dimension to be shifted, and one value for each unique track. If None,
        shifts are chosen at random.  The default is None.
    shift_dict : dictionary, optional
        Optional dictionary identifying the data points for each track.  If
        None, will be calculated. The default is None.
    update_yatc : boolean, optional
        If true, the y_atc value in D will be updated to match the shifts. The default is False.

    Returns
    -------
    shift_dict : dict
        dictionary identifying the data points for each track
    shifts : numpy array
        Shifts for each track.

    '''
    if shift_dict is not None:
        shift_keys=list(shift_dict.keys())
    else:
        shift_keys, shift_dict = \
            pc.unique_by_rows(np.c_[[getattr(D, var) for var in key_vars]].T, return_dict=True)

    if shifts is None:
        shifts = np.zeros([len(shift_keys), 2])
        for dim in dims:
            shifts[:, ['x','y'].index(dim)] = np.random.randn(len(shift_keys))*sigma

    for (this_key, ii), this_shift in zip(shift_dict.items(), shifts):
        shift_atc(D, this_shift, ind=ii)
        if update_yatc:
            D.y_atc[ii] += this_shift[1]
    return shift_dict, shifts


def apply_errors(D_noE, z_fn = None, z_args=None, sigma=None, calc_gradient=True):
    '''
    Calculate the height and errors for a data structure

    Parameters
    ----------
    D_noE : pointCollection.data
        Data structure to which errors will be added.
    z_fn : function, optional
        function that calculates the height from a data structure. The default is None.
    z_args : dict, optional
        a dictionary of arguments that will be passed to z_fn. The default is None.
    sigma : dict, optional
        a dictionary of parameter for errors.
        Default values are:
            geoloc:3.5   horizontal geolocation uncertainty
            r : 0.03 radial orbit uncertainty
            fp : 5.5 footprint width
            fp_per_seg: 57 number of footprints in each segment
            pulse_z: 0.1  pluse width expressed in range
            z_uncorr: 0.013 per-segment height uncertainty, used if the gradient is not calculated
            The default is None.
    calc_gradient : boolean, optional
        If True, the gradient of the surface is calculated and is used in the
        calculation of the per-footprint error. The default is True.

    Returns
    -------
    data : pointCollection.data
        data structure including errors.

    '''
    default_sigma = {'geoloc':3.5,
                     'r':0.03,
                     'fp_per_seg':57,
                     'fp':5.5,
                     'pulse_z':0.68*0.15,
                     'z_uncorr':0.68*0.15/np.sqrt(57)}
    temp=default_sigma.copy()
    temp.update(sigma)
    sigma=temp

    data=D_noE.copy()

    cr_dict, cr_geoloc = apply_shifts(D_noE, ['cycle','rgt'], sigma['geoloc'])

    data.assign(z=z_fn(D_noE, **z_args),
                DEM_h=z_fn(data, **z_args))

    if calc_gradient:
        temp=D_noE.copy()
        temp.x += 1
        data.assign(dz_dx = z_fn(temp, **z_args) - data.z)
        temp=D_noE.copy()
        temp.y += 1
        data.assign(dz_dy = z_fn(temp, **z_args) - data.z)
        data.assign(slope_mag = np.sqrt(data.dz_dx**2+data.dz_dy**2))

    # assign random and correlated errors
    e_corr = np.random.randn(len(cr_dict)) * sigma['r']
    for ii, ee in zip(cr_dict.values(), e_corr):
        data.z[ii] += ee

    # if we have calculated the slope magnitude, calculate the per-segment error
    # based on the surface slope, the footprint size, and the pulse width
    if 'slope_mag' in data.fields:
        if 'beam' in data.fields:
            sigma_pulse = sigma['pulse_z']/np.sqrt(4+8*data.beam)
        else:
            sigma_pulse = sigma['pulse_z']/np.sqrt(8)

        sigma_mag = np.sqrt(sigma_pulse**2 +(sigma['fp']*data.slope_mag)**2)/np.sqrt(sigma['fp_per_seg'])
        data.z += np.random.randn(data.size)*sigma_mag
        data.assign(sigma=np.zeros_like(data.x)+sigma_mag)
    else:
        data.z += np.random.randn(data.size)* sigma['z_uncorr']
        data.assign(sigma=np.zeros_like(data.x)+sigma['z_uncorr'])

    return data

def calc_along_track_slope(D, sigma_z, z_fn, **z_args):

    # calculate the along-track slope for D, given a z function and a vertical error

    z_temp=[None, None]
    for jj, dx in enumerate([-10, 10]):
        temp=D.copy()
        shift_atc(temp, [dx, 0], inplace=True)
        z_temp[jj] = z_fn(temp, **z_args)
    D.assign(dh_fit_dx = (z_temp[1]-z_temp[0])/20)
    D.assign(dh_fit_dx_sigma = sigma_z*np.sqrt(2) /20. + np.zeros_like(D.x))

def z_DEM(D, DEM=None, return_DEM=False, xy0=None, W=None, DEM_file=None, dz_of_t = None):
    '''
    read DEM data from a file and interpolate it to data points
    '''
    pad=np.array([-W/2-3300, W/2+3300])
    if DEM is None:
        DEM=pc.grid.data().from_file(DEM_file, bounds=[xy0[0]+pad, xy0[1]+pad])

    if isinstance(D, (list, tuple)):
        z=DEM.interp(*D)
    else:
        z = DEM.interp(D.x, D.y)
    if dz_of_t is None:
        if return_DEM:
            return z, DEM
        else:
            return z
    if 't' in dz_of_t:
        z += np.interp(D.t, dz_of_t.t, dz_of_t.z)
    elif 'function' in dz_of_t:
        z += dz_of_t['function'](D, **dz_of_t['args'])
    if return_DEM:
        return z, DEM
    else:
        return z


def make_sim_tracks( cycles=np.arange(1, 11), W_data = 5.e4,
                    rgt_file='RGT_001.h5', lat_0=70, lon_0=0, x_0=None, y_0=None,
                    EPSG=None,
                    product='ATL11', t_unit='days'):
    '''
    Make a simulated set of data

    Parameters
    ----------
    cycles : iterable, optional
        cycles to be included in the data. The default is np.arange(1, 11).
    W_data : float, optional
        width of the data (lat/lon square). The default is 5.e4.
    rgt_file : str, optional
        filename for a RGT data. The default is 'RGT_001.h5'.
    lat_0 : str, optional
        latitude center of the data. The default is 70.
    lon_0 : str, optional
        longitude center of the data. The default is 0.
    x_0 : str, optional
        projected x center of the data, used instead of lat_0, lon_0. The default is None.
    y_0 : TYPE, optional
        projected y center of the data, used instead of lat_0, lon_0. The default is None.
    EPSG : int, optional
        If specified, simulation results will be in projected coordinates instead
        of platte carre. The default is None.
    product : str, optional
        can be 'ATL11' or 'ATL06. The default is 'ATL11'.
    t_unit : str, optional
        time units. Can be 'seconds, 'days' or 'years. The default is 'days'.

    Returns
    -------
    pointCollection.data: data structure

    '''


    if x_0 is not None and EPSG is not None:
        temp=pc.data().from_dict({'x':np.array([x_0]), 'y':np.array([y_0])}).get_latlon(EPSG)
        lat_0 = temp.latitude[0]
        lon_0 = temp.longitude[0]

    if product=='ATL11':
        along_track_spacing=60
    else:
        along_track_spacing=20

    rgt1 = pc.data().from_h5(rgt_file, group='/')
    with h5py.File(rgt_file,'r') as h5f:
        t_orbit = h5f.attrs['t_orbit']
        # correction value
        delta_lon_orbit=h5f.attrs['delta_lon_orbit'] +-0.00096
    rgt1.assign(x_atc=calc_xatc(rgt1))

    lat_tol = np.maximum(W_data+3300+7000, 7000)/(6378e3*2*np.pi/360.)
    lon_tol = np.maximum(W_data+3300+7000, 10000)/(6378e3*2*np.pi/360.*np.cos(lat_0*np.pi/180))

    # pick the parts of the orbit that are ascending and descending
    desc = (rgt1.t > t_orbit/4) & (rgt1.t < 3*t_orbit/4)
    asc = desc==0
    rgt_desc = rgt1[desc & (np.abs(rgt1.latitude - lat_0) < lat_tol)]
    rgt_asc = rgt1[asc & (np.abs(rgt1.latitude - lat_0) < lat_tol)]

    orbs={key:[] for key in ['asc','desc']}

    # replicate the rgt for each orbit in a cycle, keep data close to the reference point
    for track in range(1387):
        for key, rgt in zip(['asc','desc'], [rgt_asc, rgt_desc]):
            temp=rgt.copy()
            temp.t += t_orbit*track
            temp.longitude += delta_lon_orbit * track
            delta_lon = np.mod(temp.longitude-lon_0+180, 360)-180
            delta_lat = temp.latitude-lat_0
            keep = (np.abs(delta_lon) < lon_tol) & (np.abs(delta_lat)<lat_tol)
            if np.any(keep):
                temp=temp[keep]
                temp.longitude = np.mod(temp.longitude+180, 360)-180
                temp.assign(rgt=np.zeros_like(temp.t)+track)
                orbs[key] += [temp]

    rgts=[]
    for orb in orbs['asc']+orbs['desc']:
        if EPSG is None:
            x, y = to_xy(orb.latitude, orb.longitude, lat_0, lon_0)
            orb.assign(x=x, y=y)
        else:
            orb.get_xy(EPSG=EPSG)
        segment_ids_float = orb.x_atc/along_track_spacing
        segment_ids = np.arange(np.ceil(segment_ids_float[0]), np.floor(segment_ids_float[-1]))
        rgt=pc.data().from_dict({field:np.interp(segment_ids, segment_ids_float, var)
                                 for field, var in zip(['x','y','t','rgt', 'x_atc', 'longitude','latitude'],
                                                       [orb.x, orb.y, orb.t, orb.rgt, orb.x_atc, orb.longitude, orb.latitude])})
        rgt.assign(y_atc=np.zeros_like(rgt.x))
        rgt.assign(segment_id = segment_ids * int(along_track_spacing/20))
        # shift by zero so that the along-track vectors are defined.
        shift_atc(rgt, [0, 0])
        calc_azimuth(rgt, lat_0, lon_0)
        rgts += [rgt]

    # shift the rgts to make the beam pairs
    D_GTs=[]
    for rgt in rgts:
        for pair, dy in zip([1, 2, 3], [-3.3e3, 0, 3.3e3]):
            D_GT=rgt.copy()
            #print([rgt.rgt[0], pair, 'before', np.min(D_GT.x), np.max(D_GT.x)])
            shift_atc(D_GT, [0, dy])
            D_GT.y_atc += dy
            #print([rgt.rgt[0], pair, 'after', np.min(D_GT.x), np.max(D_GT.x)])
            D_GT.assign(pair=pair+np.zeros_like(D_GT.t))
            if product=='ATL11':
                D_GTs += [D_GT]
            else:
                # if we're making ATL06, shift the pairs to make the beams
                for beam, dyb in zip([0, 1], [-45, 45]):
                    D_beam = D_GT.copy()
                    shift_atc(D_beam, [0, dyb])
                    D_beam.y_atc += dyb
                    D_beam.assign(beam = beam+np.zeros_like(D_beam.t))

                    D_GTs += [D_beam]
            #print()
    D_cycle1=pc.data().from_list(D_GTs)

    if EPSG is None:
        x0=0
        y0=0
    else:
        temp=pc.data().from_dict({'latitude':np.array([lat_0]),
                                  'longitude':np.array([lon_0])}).get_xy(EPSG=EPSG)
        x0, y0 = [temp.x, temp.y]

    D_cycle1.index(np.all(np.abs(np.c_[D_cycle1.x-x0, D_cycle1.y-y0]) < W_data/2+200, axis=1))

    # replicate cycle1 to make a full set of cycles
    D=[]
    for cycle in cycles:
        D_cycle=D_cycle1.copy()
        D_cycle.t += 91*(cycle-1)
        D_cycle.assign(cycle=np.zeros_like(D_cycle.t)+cycle)
        D += [D_cycle]
    D=pc.data().from_list(D)

    if EPSG is not None:
        D.get_latlon(EPSG=EPSG)

    # time unit is days, can convert to seconds or years
    if t_unit=='year' or t_unit=='years' or t_unit == 'yr':
        D.t /= 365.25
    elif t_unit=='second' or t_unit=='seconds' or t_unit=='s':
        D.t *= 24*3600
    return D


# testing code:


if False:
    xy0 = [-40000, -2000000]
    D=make_sim_tracks(EPSG=3413, x_0=xy0[0], y_0=xy0[1], product='ATL06', t_unit='year')
    apply_shifts(
        D,
        ['rgt','cycle'], 10, update_yatc=True);
    def z_plane(D, mean_slope=0.1, xy0=[0,0] ):
        return (D.x-xy0[0])*mean_slope
    z_fn = z_plane
    z_args = {'mean_slope':0.1, 'xy0':xy0}
    sigma={'geoloc':3.5,'r':0.03,'z_corr':0.03,'z_uncorr':0.15/np.sqrt(57)}
    data=apply_errors(D, z_fn = z_plane, z_args=z_args, sigma=sigma)

    from LSsurf.smooth_fit import smooth_fit
    E_d3zdx2dt=0.0001
    E_d2z0dx2=0.06
    E_d2zdt2=5000

    cycle_range=[np.min(data.cycle), np.max(data.cycle)]
    bias_params=None
    data.assign({'sigma':np.zeros_like(data.z)+0.1})

    data_nobias=data.copy()

    ctr={'x':xy0[0],'y':xy0[1],'t':np.mean(cycle_range)*0.25}
    W={'x':2.e4,'y':2.e4,'t':np.diff(cycle_range)*0.25}
    spacing={'z0':100, 'dz':500, 'dt':0.25}

    data_gap_scale=2500
    E_RMS={'d2z0_dx2':E_d2z0dx2, 'dz0_dx':E_d2z0dx2*data_gap_scale, 'd3z_dx2dt':E_d3zdx2dt, 'd2z_dxdt':E_d3zdx2dt*data_gap_scale,  'd2z_dt2':E_d2zdt2}

    # run the fit with no bias params
    S_nobias=smooth_fit(data=data_nobias[::5], ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS,
                 reference_epoch=5, compute_E=False,
                 max_iterations=1,
                 bias_params=None,
                 VERBOSE=True, dzdt_lags=[1])

if False:
    bias_params=['cycle','rgt']
    data.assign({'sigma_corr':np.zeros_like(data.z)+((z_args['mean_slope']*sigma['geoloc'])**2+0.03**2)**0.5})

    # run the fit with bias params
    S_bias=smooth_fit(data=data[::5], ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS,
                 reference_epoch=5, compute_E=False,
                 max_iterations=1,
                 bias_params=bias_params,
                 VERBOSE=True, dzdt_lags=[1])
