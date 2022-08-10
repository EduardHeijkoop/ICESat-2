import numpy as np
import datetime
import h5py
from icesat2_utils import gps2utc,great_circle_distance
import scipy

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.read_FES_model import extract_FES_constants


total_seconds = np.vectorize(datetime.timedelta.total_seconds)

def analyze_icesat2_ocean(icesat2_dir,df_city,model_dir,geophys_corr_toggle=True,ocean_tide_replacement_toggle=False,extrapolate_fes2014=True):
    #Given a directory of downloaded ATL03 hdf5 files,
    #reads them and writes the high confidence photons to a CSV as:
    #longitude,latitude,height (WGS84),time [UTC]
    city_name = df_city.city
    icesat2_list = icesat2_dir+city_name + '/icesat2_list.txt'
    with open(icesat2_list) as f3:
        file_list = f3.read().splitlines()
    beam_list_r = ['gt1r','gt2r','gt3r']
    beam_list_l = ['gt1l','gt2l','gt3l']
    #Initialize arrays and start reading .h5 files
    #lon = np.empty([0,1],dtype=float)
    #lat = np.empty([0,1],dtype=float)
    #h = np.empty([0,1],dtype=float)
    lon_high_med_conf = np.empty([0,1],dtype=float)
    lat_high_med_conf = np.empty([0,1],dtype=float)
    h_high_med_conf = np.empty([0,1],dtype=float)
    delta_time_total_high_med_conf = np.empty([0,1],dtype=float)
    for h5_file in file_list:
        full_file = icesat2_dir + city_name + '/' + h5_file
        atl03_file = h5py.File(full_file,'r')
        #list(atl03_file.keys())
        sc_orient = atl03_file['/orbit_info/sc_orient'][0]
        #Select strong beams according to S/C orientation
        if sc_orient == 1:
            beam_list_req = beam_list_r
        elif sc_orient == 0:
            beam_list_req = beam_list_l
        elif sc_orient == 2:
            continue
        for beam in beam_list_req:
            #Some beams don't actually have any height data in them, so this is done to skip those
            #Sometimes only one or two beams are present, this also prevents looking for those
            heights_check = False
            heights_check = '/'+beam+'/heights' in atl03_file
            if heights_check == False:
                continue
            tmp_sdp = np.asarray(atl03_file['/ancillary_data/atlas_sdp_gps_epoch']).squeeze()
            #full photon rate
            tmp_lon = np.asarray(atl03_file['/'+beam+'/heights/lon_ph']).squeeze()
            tmp_lat = np.asarray(atl03_file['/'+beam+'/heights/lat_ph']).squeeze()
            tmp_h = np.asarray(atl03_file['/'+beam+'/heights/h_ph']).squeeze()
            tmp_delta_time = np.asarray(atl03_file['/'+beam+'/heights/delta_time']).squeeze()
            tmp_delta_time_total = tmp_sdp + tmp_delta_time
            tmp_signal_conf = np.asarray(atl03_file['/'+beam+'/heights/signal_conf_ph'])
            tmp_high_med_conf = np.logical_or(tmp_signal_conf[:,1]==3,tmp_signal_conf[:,1]==4)
            #If fewer than 100 high/medium confidence photons are in an hdf5 file, skip it
            if np.sum(tmp_high_med_conf) < 100:
                continue
            #reference photon rate
            tmp_lon_ref = np.asarray(atl03_file['/'+beam+'/geolocation/reference_photon_lon']).squeeze()
            tmp_lat_ref = np.asarray(atl03_file['/'+beam+'/geolocation/reference_photon_lat']).squeeze()
            tmp_delta_time_geophys_corr = np.asarray(atl03_file['/'+beam+'/geophys_corr/delta_time']).squeeze()
            tmp_delta_time_total_geophys_corr = tmp_sdp + tmp_delta_time_geophys_corr
            tmp_ph_index_beg = np.asarray(atl03_file['/'+beam+'/geolocation/ph_index_beg']).squeeze()
            tmp_ph_index_beg = tmp_ph_index_beg - 1
            tmp_segment_ph_cnt = np.asarray(atl03_file['/'+beam+'/geolocation/segment_ph_cnt']).squeeze()
            tmp_ref_ph_index = np.asarray(atl03_file['/'+beam+'/geolocation/reference_photon_index']).squeeze()
            tmp_ph_index_end = tmp_ph_index_beg + tmp_segment_ph_cnt
            tmp_ocean_tide = np.asarray(atl03_file['/'+beam+'/geophys_corr/tide_ocean']).squeeze()
            tmp_dac = np.asarray(atl03_file['/'+beam+'/geophys_corr/dac']).squeeze()
            tmp_eq_tide = np.asarray(atl03_file['/'+beam+'/geophys_corr/tide_equilibrium']).squeeze()
            tmp_podppd_flag = np.asarray(atl03_file['/'+beam+'/geolocation/podppd_flag']).squeeze()
            #no valid photons to "create" a ref photon -> revert back to reference ground track, which we don't want, so select segments with >0 photons
            idx_ref_ph = tmp_segment_ph_cnt>0

            tmp_lon_ref = tmp_lon_ref[idx_ref_ph]
            tmp_lat_ref = tmp_lat_ref[idx_ref_ph]
            tmp_delta_time_total_geophys_corr = tmp_delta_time_total_geophys_corr[idx_ref_ph]
            tmp_ph_index_beg = tmp_ph_index_beg[idx_ref_ph]
            tmp_ph_index_end = tmp_ph_index_end[idx_ref_ph]
            tmp_segment_ph_cnt = tmp_segment_ph_cnt[idx_ref_ph]
            tmp_ref_ph_index = tmp_ref_ph_index[idx_ref_ph]
            tmp_ocean_tide = tmp_ocean_tide[idx_ref_ph]
            tmp_dac = tmp_dac[idx_ref_ph]
            tmp_eq_tide = tmp_eq_tide[idx_ref_ph]
            tmp_podppd_flag = tmp_podppd_flag[idx_ref_ph]
            tmp_podppd_flag_full_ph = np.zeros(tmp_lon.shape)
            for i in range(len(tmp_ph_index_beg)):
                tmp_podppd_flag_full_ph[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] = tmp_podppd_flag[i]
            tmp_idx_podppd = tmp_podppd_flag_full_ph == 0


            if geophys_corr_toggle == True:
                #even after ref photon filtering, still areas with no tides (due to 0.5x0.5deg resolution)
                #also no equilibrium tides there, if we are going to apply geophysical corrections without
                #tide model swapping, skip the areas with no tides

                #not all photons associated with a particular reference photon have the same signal confidences,
                #i.e. do the high/med conf filtering after all the reference photon rate analysis has been done
                #to prevent indexing problems
                #Note: by convention we *subtract* the corrections to apply them
                if ocean_tide_replacement_toggle == False:
                    idx_no_got_tides = tmp_ocean_tide > 1e38
                    for i in np.atleast_1d(np.argwhere(idx_no_got_tides==False).squeeze()):
                        tmp_h[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] -= (tmp_ocean_tide[i] + tmp_dac[i] + tmp_eq_tide[i])
                    for i in np.atleast_1d(np.argwhere(idx_no_got_tides).squeeze()):
                        tmp_h[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] = np.nan
                else:
                    tmp_utc_time_geophys_corr = gps2utc(tmp_delta_time_total_geophys_corr)
                    fes2014_heights = ocean_tide_replacement(tmp_lon_ref,tmp_lat_ref,tmp_utc_time_geophys_corr,model_dir)
                    idx_no_fes_tides = np.isnan(fes2014_heights)
                    if np.sum(~idx_no_fes_tides) < 100:
                        continue
                    dist_ref_ph = great_circle_distance(tmp_lon_ref,tmp_lat_ref,tmp_lon_ref[0],tmp_lat_ref[0])
                    if extrapolate_fes2014 == True:
                        interp_func = scipy.interpolate.interp1d(dist_ref_ph[~idx_no_fes_tides],fes2014_heights[~idx_no_fes_tides],kind='cubic',fill_value='extrapolate')
                    else:
                        interp_func = scipy.interpolate.interp1d(dist_ref_ph[~idx_no_fes_tides],fes2014_heights[~idx_no_fes_tides],kind='cubic')
                    fes_interp = interp_func(dist_ref_ph)
                    for i in range(len(tmp_ref_ph_index)):
                        tmp_h[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] -= (fes_interp[i] + tmp_dac[i])

                    # for i in np.atleast_1d(np.argwhere(idx_no_fes_tides==False).squeeze()):
                    #     tmp_h[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] -= (fes2014_heights[i] + tmp_dac[i])
                    # for i in np.atleast_1d(np.argwhere(idx_no_fes_tides).squeeze()):
                    #     tmp_h[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] = np.nan

            idx_nan = np.isnan(tmp_h)
            idx_flags = np.all((tmp_high_med_conf,tmp_idx_podppd,~idx_nan),axis=0)

            tmp_lon_high_med_conf = tmp_lon[idx_flags]
            tmp_lat_high_med_conf = tmp_lat[idx_flags]
            tmp_h_high_med_conf = tmp_h[idx_flags]
            tmp_delta_time_total_high_med_conf = tmp_delta_time_total[idx_flags]

            lon_high_med_conf = np.append(lon_high_med_conf,tmp_lon_high_med_conf)
            lat_high_med_conf = np.append(lat_high_med_conf,tmp_lat_high_med_conf)
            h_high_med_conf = np.append(h_high_med_conf,tmp_h_high_med_conf)
            delta_time_total_high_med_conf = np.append(delta_time_total_high_med_conf,tmp_delta_time_total_high_med_conf)

    return lon_high_med_conf,lat_high_med_conf,h_high_med_conf,delta_time_total_high_med_conf

def ocean_tide_replacement(lon,lat,utc_time,model_dir):
    '''
    #Given a set of lon,lat and utc time, computes FES2014 tidal elevations
    '''
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    time_datetime = np.asarray(list(map(datetime.datetime.fromisoformat,utc_time)))
    unique_date_list = np.unique([a.date() for a in time_datetime])
    tide_heights = np.empty(len(lon),dtype=np.float32)
    for unique_date in unique_date_list:
        idx_unique_date = np.asarray([a.date() == unique_date for a in time_datetime])
        time_unique_date = time_datetime[idx_unique_date]
        lon_unique_date = lon[idx_unique_date]
        lat_unique_date = lat[idx_unique_date]
        YMD = time_unique_date[0].date()
        unique_seconds = np.unique(np.asarray([a.hour*3600+a.minute*60+a.second for a in time_unique_date]))
        seconds = np.arange(np.min(unique_seconds),np.max(unique_seconds)+2)
        seconds_since_midnight = [a.hour*3600 + a.minute*60 + a.second + a.microsecond/1000000 for a in time_unique_date]
        idx_time = np.asarray([np.argmin(abs(t - seconds)) for t in seconds_since_midnight])
        tide_time = pyTMD.time.convert_calendar_dates(YMD.year,YMD.month,YMD.day,second=seconds)
        amp,ph = extract_FES_constants(np.atleast_1d(lon_unique_date),
                np.atleast_1d(lat_unique_date), model.model_file, TYPE=model.type,
                VERSION=model.version, METHOD='spline', EXTRAPOLATE=False,
                SCALE=model.scale, GZIP=model.compressed)
        DELTAT = calc_delta_time(delta_file, tide_time)
        cph = -1j*ph*np.pi/180.0
        hc = amp*np.exp(cph)
        tmp_tide_heights = np.empty(len(lon_unique_date))
        for i in range(len(lon_unique_date)):
            if np.any(amp[i].mask) == True:
                tmp_tide_heights[i] = np.nan
            else:
                TIDE =         predict_tidal_ts(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,DELTAT=DELTAT[idx_time[i]],CORRECTIONS=model.format)
                MINOR = infer_minor_corrections(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,DELTAT=DELTAT[idx_time[i]],CORRECTIONS=model.format)
                TIDE.data[:] += MINOR.data[:]
                tmp_tide_heights[i] = TIDE.data
        tide_heights[idx_unique_date] = tmp_tide_heights
    return tide_heights

def cluster_photon_cloud(lon,lat,time,height,clustering_method='DBSCAN'):
    '''
    Clusters photon cloud with either DBSCAN or 3sigma along-track
    '''
    if clustering_method == 'DBSCAN':
        
        return None
    elif clustering_method == '3sigma':
        return None