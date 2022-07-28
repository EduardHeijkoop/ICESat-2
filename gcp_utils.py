import numpy as np
import h5py
from icesat2_utils import gps2utc


def analyze_icesat2_land(icesat2_dir,df_city,shp_data):
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
    # lon = np.empty([0,1],dtype=float)
    # lat = np.empty([0,1],dtype=float)
    # h = np.empty([0,1],dtype=float)
    lon_high_conf = np.empty([0,1],dtype=float)
    lat_high_conf = np.empty([0,1],dtype=float)
    h_high_conf = np.empty([0,1],dtype=float)
    delta_time_total_high_conf = np.empty([0,1],dtype=float)
    for h5_file in file_list:
        full_file = icesat2_dir + city_name + '/' + h5_file
        atl03_file = h5py.File(full_file,'r')
        list(atl03_file.keys())
        sc_orient = atl03_file['/orbit_info/sc_orient']
        sc_orient = sc_orient[0]
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
            tmp_lon = np.asarray(atl03_file['/'+beam+'/heights/lon_ph']).squeeze()
            tmp_lat = np.asarray(atl03_file['/'+beam+'/heights/lat_ph']).squeeze()
            tmp_h = np.asarray(atl03_file['/'+beam+'/heights/h_ph']).squeeze()
            tmp_sdp = np.asarray(atl03_file['/ancillary_data/atlas_sdp_gps_epoch']).squeeze()
            tmp_delta_time = np.asarray(atl03_file['/'+beam+'/heights/delta_time']).squeeze()
            tmp_delta_time_total = tmp_sdp + tmp_delta_time
            tmp_signal_conf = np.asarray(atl03_file['/'+beam+'/heights/signal_conf_ph'])
            tmp_high_conf = tmp_signal_conf[:,0] == 4
            #If fewer than 100 high confidence photons are in an hdf5 file, skip
            if len(tmp_high_conf) < 100:
                continue
            tmp_ph_index_beg = np.asarray(atl03_file['/'+beam+'/geolocation/ph_index_beg']).squeeze()
            tmp_ph_index_beg = tmp_ph_index_beg - 1
            tmp_segment_ph_cnt = np.asarray(atl03_file['/'+beam+'/geolocation/segment_ph_cnt']).squeeze()
            tmp_ref_ph_index = np.asarray(atl03_file['/'+beam+'/geolocation/reference_photon_index']).squeeze()
            tmp_ph_index_end = tmp_ph_index_beg + tmp_segment_ph_cnt
            tmp_podppd_flag = np.asarray(atl03_file['/'+beam+'/geolocation/podppd_flag']).squeeze()
            #no valid photons to "create" a ref photon -> revert back to reference ground track, which we don't want, so select segments with >0 photons
            idx_ref_ph = tmp_segment_ph_cnt>0
            tmp_ph_index_beg = tmp_ph_index_beg[idx_ref_ph]
            tmp_ph_index_end = tmp_ph_index_end[idx_ref_ph]
            tmp_segment_ph_cnt = tmp_segment_ph_cnt[idx_ref_ph]
            tmp_ref_ph_index = tmp_ref_ph_index[idx_ref_ph]
            tmp_podppd_flag = tmp_podppd_flag[idx_ref_ph]
            tmp_podppd_flag_full_ph = np.zeros(tmp_lon.shape)
            for i in range(len(tmp_ph_index_beg)):
                tmp_podppd_flag_full_ph[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] = tmp_podppd_flag[i]
            tmp_idx_podppd = tmp_podppd_flag_full_ph == 0

            idx_nan = np.isnan(tmp_h)
            idx_flags = np.all((tmp_high_conf,tmp_idx_podppd,~idx_nan),axis=0)

            tmp_lon_high_conf = tmp_lon[idx_flags]
            tmp_lat_high_conf = tmp_lat[idx_flags]
            tmp_h_high_conf = tmp_h[idx_flags]
            tmp_delta_time_total_high_conf = tmp_delta_time_total[idx_flags]
            lon_high_conf = np.append(lon_high_conf,tmp_lon_high_conf)
            lat_high_conf = np.append(lat_high_conf,tmp_lat_high_conf)
            h_high_conf = np.append(h_high_conf,tmp_h_high_conf)
            delta_time_total_high_conf = np.append(delta_time_total_high_conf,tmp_delta_time_total_high_conf)
    #A lot of data will be captured off the coast that we don't want, this is a quick way of getting rid of that
    #Also prevents areas with no SRTM from being queried
    idx_lon = np.logical_or(lon_high_conf < np.min(shp_data.bounds.minx),lon_high_conf > np.max(shp_data.bounds.maxx))
    idx_lat = np.logical_or(lat_high_conf < np.min(shp_data.bounds.miny),lat_high_conf > np.max(shp_data.bounds.maxy))
    idx_tot = np.logical_or(idx_lon,idx_lat)
    lon_high_conf = lon_high_conf[~idx_tot]
    lat_high_conf = lat_high_conf[~idx_tot]
    h_high_conf = h_high_conf[~idx_tot]
    delta_time_total_high_conf = delta_time_total_high_conf[~idx_tot]
    return lon_high_conf,lat_high_conf,h_high_conf,delta_time_total_high_conf
