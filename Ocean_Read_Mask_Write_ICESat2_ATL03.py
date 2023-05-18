import os
import numpy as np
import pandas as pd
import datetime
import configparser
import warnings
import argparse

from icesat2_utils import get_token,get_osm_extents,create_bbox,move_icesat2,download_icesat2
from icesat2_utils import gps2utc,landmask_icesat2,DTU21_filter_icesat2
from ocean_utils import analyze_icesat2_ocean

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###
#Update March 2021: now does landmasking in C
#Update December 2021: rewrite into functions
#Update February 2022: add ability to correct using FES2014 instead of standard GOT4.8
#Update May 2022: Moved all functions to icesat2_utils.py for harmonization with GCP

#This script will download ICESat-2 ATL03 geolocated photons for a given region.
#The point cloud will be masked with a given shapefile (e.g. a coastline), originally used as ground control points (GCPs)
#Output is a .txt file with ICESat-2 ATL03 data in the format:
#Longitude [deg], Latitude [deg], Height [m above WGS84], Time [UTC]


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'icesat2_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    parser.add_argument('--beams',action='store_true',default=False,help='Toggle to print beams.')
    args = parser.parse_args()
    machine_name = args.machine
    beam_flag = args.beams

    DTU21_toggle = config.getboolean('OCEAN_CONSTANTS','DTU21_toggle')
    landmask_toggle = config.getboolean('OCEAN_CONSTANTS','landmask_toggle')
    timestamp_toggle = config.getboolean('OCEAN_CONSTANTS','timestamp_toggle')
    geophys_corr_toggle = config.getboolean('OCEAN_CONSTANTS','geophys_corr_toggle')
    ocean_tide_replacement_toggle = config.getboolean('OCEAN_CONSTANTS','ocean_tide_replacement_toggle')
    clustering_toggle = config.getboolean('OCEAN_CONSTANTS','clustering_toggle')
    on_off_str = ('off','on')

    print('Current settings:')
    print(f'DTU21 filtering         : {on_off_str[DTU21_toggle]}')
    print(f'Landmask                : {on_off_str[landmask_toggle]}')
    print(f'Timestamps              : {on_off_str[timestamp_toggle]}')
    print(f'Geophysical corrections : {on_off_str[geophys_corr_toggle]}')
    print(f'Ocean tide replacement  : {on_off_str[ocean_tide_replacement_toggle]}')
    print(f'Clustering              : {on_off_str[clustering_toggle]}')

    input_file = config.get('OCEAN_PATHS','input_file') #Input file with location name,lon_min,lon_max,lat_min,lat_max (1 header line)
    osm_shp_path = config.get('GENERAL_PATHS','osm_shp_path') #OpenStreetMap land polygons, available at https://osmdata.openstreetmap.de/data/land-polygons.html (use WGS84, not split)
    icesat2_dir = config.get('OCEAN_PATHS','icesat2_dir') #output directory, which will be populated by subdirectories named after your input
    error_log_file = config.get('OCEAN_PATHS','error_log_file') #file to write errors to
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file') #file with C function pnpoly, "point in polygon", to perform landmask
    landmask_inside_flag = config.getint('OCEAN_CONSTANTS','landmask_inside_flag') #flag to find points inside (1 for land) or outside (0 for water) polygon
    model_dir = config.get('OCEAN_PATHS','model_dir')

    user = config.get('GENERAL','user') #Your NASA EarthData username
    token = get_token(user) #Create NSIDC token to download ICESat-2
    if DTU21_toggle == True:
        DTU21_threshold = config.getfloat('OCEAN_CONSTANTS','DTU21_threshold')
        DTU21_threshold_str = str(DTU21_threshold).replace('.','p')
        DTU21_path = config.get('OCEAN_PATHS','DTU21_path') #path to DTU21 file

    if machine_name == 'b':
        osm_shp_path = osm_shp_path.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        if DTU21_toggle:
            DTU21_path = DTU21_path.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        
    elif machine_name == 'local':
        osm_shp_path = osm_shp_path.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/Sea_Level/','/media/heijkoop/DATA/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/Sea_Level/','/media/heijkoop/DATA/')
        landmask_c_file = landmask_c_file.replace('/home/eheijkoop/Scripts/','/media/heijkoop/DATA/Dropbox/TU/PhD/Github/')
        if DTU21_toggle:
            DTU21_path = DTU21_path.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')

    if not os.path.isdir(icesat2_dir):
        os.mkdir(icesat2_dir)

    df_extents = pd.read_csv(input_file,header=0,names=[
        'city',
        'lon_min',
        'lon_max',
        'lat_min',
        'lat_max',
        't_start',
        't_end'],
        dtype={
            'city':'str',
            'lon_min':'float',
            'lon_max':'float',
            'lat_min':'float',
            'lat_max':'float',
            't_start':'str',
            't_end':'str'}
        )
    for i in range(len(df_extents)):
        city_name = df_extents.city[i]
        if city_name == 'Break':
            break
        print('Working on ' + city_name)
        if not os.path.isdir(icesat2_dir+city_name):
            os.mkdir(icesat2_dir + city_name)
        lon_coast,lat_coast,shp_data = get_osm_extents(df_extents.iloc[i],osm_shp_path,icesat2_dir)
        bbox_code = create_bbox(icesat2_dir,df_extents.iloc[i])
        if bbox_code is not None:
            continue
        download_code = download_icesat2(df_extents.iloc[i],token,error_log_file,version=5)
        if download_code is not None:
            continue
        move_code = move_icesat2(icesat2_dir,df_extents.iloc[i])
        if move_code is not None:
            continue
        if beam_flag == True:
            lon_high_med_conf,lat_high_med_conf,h_high_med_conf,delta_time_total_high_med_conf,beam_high_med_conf = analyze_icesat2_ocean(icesat2_dir,df_extents.iloc[i],model_dir,geophys_corr_toggle,ocean_tide_replacement_toggle,beam_flag)
        else:    
            lon_high_med_conf,lat_high_med_conf,h_high_med_conf,delta_time_total_high_med_conf = analyze_icesat2_ocean(icesat2_dir,df_extents.iloc[i],model_dir,geophys_corr_toggle,ocean_tide_replacement_toggle,beam_flag)
        if len(lon_high_med_conf) == 0:
            continue
        if landmask_toggle == True:
            landmask = landmask_icesat2(lon_high_med_conf,lat_high_med_conf,lon_coast,lat_coast,landmask_c_file,landmask_inside_flag)
            lon_high_med_conf = lon_high_med_conf[landmask]
            lat_high_med_conf = lat_high_med_conf[landmask]
            h_high_med_conf = h_high_med_conf[landmask]
            delta_time_total_high_med_conf = delta_time_total_high_med_conf[landmask]
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_med_conf_masked.txt'
            if beam_flag == True:
                beam_high_med_conf = beam_high_med_conf[landmask]
        else:
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_med_conf.txt'
        utc_time_high_med_conf = gps2utc(delta_time_total_high_med_conf)

        if geophys_corr_toggle == False:
            icesat2_file = icesat2_file.replace('ATL03','UNCORRECTED_ATL03')
        if ocean_tide_replacement_toggle == True:
            icesat2_file = icesat2_file.replace('_ATL03','_ATL03_FES2014')
        
        if timestamp_toggle == True:
            if beam_flag == True:
                np.savetxt(icesat2_file,np.c_[lon_high_med_conf,lat_high_med_conf,h_high_med_conf,utc_time_high_med_conf.astype(object),beam_high_med_conf.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s,%s',delimiter=',')
            else:
                np.savetxt(icesat2_file,np.c_[lon_high_med_conf,lat_high_med_conf,h_high_med_conf,utc_time_high_med_conf.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s',delimiter=',')
        else:
            if beam_flag == True:
                np.savetxt(icesat2_file,np.c_[lon_high_med_conf,lat_high_med_conf,h_high_med_conf,beam_high_med_conf.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s',delimiter=',')
            else:
                np.savetxt(icesat2_file,np.c_[lon_high_med_conf,lat_high_med_conf,h_high_med_conf],fmt='%11.6f,%11.6f,%11.6f',delimiter=',')
        
        if DTU21_toggle == True:
            DTU21_cond = DTU21_filter_icesat2(h_high_med_conf,icesat2_file,icesat2_dir,df_extents.iloc[i],DTU21_threshold,DTU21_path)
            lon_high_med_conf_DTU21 = lon_high_med_conf[DTU21_cond]
            lat_high_med_conf_DTU21 = lat_high_med_conf[DTU21_cond]
            h_high_med_conf_DTU21 = h_high_med_conf[DTU21_cond]
            delta_time_total_high_med_conf_DTU21 = delta_time_total_high_med_conf[DTU21_cond]
            utc_time_high_med_conf_DTU21 = gps2utc(delta_time_total_high_med_conf_DTU21)
            beam_high_med_conf_DTU21 = beam_high_med_conf[DTU21_cond]
            if landmask_toggle == True:
                icesat2_dtu21_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_med_conf_masked_DTU21_filtered_threshold_{DTU21_threshold_str}_m.txt'
            else:
                icesat2_dtu21_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_med_conf_DTU21_filtered_threshold_{DTU21_threshold_str}_m.txt'
            if geophys_corr_toggle == False:
                icesat2_dtu21_file = icesat2_dtu21_file.replace('ATL03','UNCORRECTED_ATL03')
            if ocean_tide_replacement_toggle == True:
                icesat2_dtu21_file = icesat2_dtu21_file.replace('_ATL03','_ATL03_FES2014')
            if timestamp_toggle == True:
                if beam_flag == True:
                    np.savetxt(icesat2_dtu21_file,np.c_[lon_high_med_conf_DTU21,lat_high_med_conf_DTU21,h_high_med_conf_DTU21,utc_time_high_med_conf_DTU21.astype(object),beam_high_med_conf_DTU21.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s,%s',delimiter=',')
                else:
                    np.savetxt(icesat2_dtu21_file,np.c_[lon_high_med_conf_DTU21,lat_high_med_conf_DTU21,h_high_med_conf_DTU21,utc_time_high_med_conf_DTU21.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s',delimiter=',')
            else:
                if beam_flag == True:
                    np.savetxt(icesat2_dtu21_file,np.c_[lon_high_med_conf_DTU21,lat_high_med_conf_DTU21,h_high_med_conf_DTU21,beam_high_med_conf_DTU21.astype(object)],fmt='%11.6f,%11.6f,%11.6f,%s',delimiter=',')
                else:
                    np.savetxt(icesat2_dtu21_file,np.c_[lon_high_med_conf_DTU21,lat_high_med_conf_DTU21,h_high_med_conf_DTU21],fmt='%11.6f,%11.6f,%11.6f',delimiter=',')
        print(f'Done with {city_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(' ')

if __name__ == '__main__':
    main()
