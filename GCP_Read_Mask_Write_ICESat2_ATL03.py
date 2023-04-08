import os
import numpy as np
import pandas as pd
import datetime
import getpass
import configparser
import argparse
import warnings

from icesat2_utils import get_token,get_osm_extents,create_bbox,move_icesat2,download_icesat2
from icesat2_utils import gps2utc,landmask_icesat2,SRTM_filter_icesat2
from gcp_utils import analyze_icesat2_land

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###
#Update March 2021: now does landmasking in C
#Update December 2021: rewrite into functions
#Update May 2022: Moved all functions to icesat2_utils.py for harmonization with ocean
#                 Added constants/paths to icesat2_config.ini file

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
    parser.add_argument('--weak',action='store_true',default=False,help='Toggle to analyze weak beams.')
    args = parser.parse_args()
    machine_name = args.machine
    beam_flag = args.beams
    weak_flag = args.weak

    SRTM_toggle = config.getboolean('GCP_CONSTANTS','SRTM_toggle')
    landmask_toggle = config.getboolean('GCP_CONSTANTS','landmask_toggle')
    timestamp_toggle = config.getboolean('GCP_CONSTANTS','timestamp_toggle')
    on_off_str = ('off','on')

    print('Current settings:')
    print(f'SRTM filtering : {on_off_str[SRTM_toggle]}')
    print(f'Landmask       : {on_off_str[landmask_toggle]}')
    print(f'Timestamps     : {on_off_str[timestamp_toggle]}')

    input_file = config.get('GCP_PATHS','input_file') #Input file with location name,lon_min,lon_max,lat_min,lat_max (1 header line)
    osm_shp_path = config.get('GENERAL_PATHS','osm_shp_path') #OpenStreetMap land polygons, available at https://osmdata.openstreetmap.de/data/land-polygons.html (use WGS84, not split)
    icesat2_dir = config.get('GCP_PATHS','icesat2_dir') #output directory, which will be populated by subdirectories named after your input
    error_log_file = config.get('GCP_PATHS','error_log_file') #file to write errors to
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file') #file with C function pnpoly, "point in polygon", to perform landmask
    landmask_inside_flag = config.getint('GCP_CONSTANTS','landmask_inside_flag') #flag to find points inside (1) or outside (0) polygon

    user = config.get('GENERAL','user') #Your NASA EarthData username
    token = get_token(user) #Create NSIDC token to download ICESat-2
    if SRTM_toggle:
        pw = getpass.getpass() #Your NASA EarthData password
        SRTM_threshold = config.getfloat('GCP_CONSTANTS','SRTM_Threshold') #set your SRTM threshold here
        SRTM_threshold_str = str(SRTM_threshold).replace('.','p') #replace decimal point with p for file name
        EGM96_path = config.get('GCP_PATHS','EGM96_path') #supplied on github

    if machine_name == 'b':
        osm_shp_path = osm_shp_path.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        if SRTM_toggle:
            EGM96_path = EGM96_path.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        
    elif machine_name == 'local':
        osm_shp_path = osm_shp_path.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/','/media/heijkoop/DATA/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/','/media/heijkoop/DATA/')
        landmask_c_file = landmask_c_file.replace('/home/eheijkoop/Scripts/','/media/heijkoop/DATA/Dropbox/TU/PhD/Github/')
        if SRTM_toggle:
            EGM96_path = EGM96_path.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/GEOID/')
            
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
        download_code = download_icesat2(df_extents.iloc[i],token,error_log_file)
        if download_code is not None:
            continue
        move_code = move_icesat2(icesat2_dir,df_extents.iloc[i])
        if move_code is not None:
            continue
        lon_high_conf,lat_high_conf,h_high_conf,delta_time_total_high_conf,beam_high_conf = analyze_icesat2_land(icesat2_dir,df_extents.iloc[i],shp_data,beam_flag,weak_flag)
        if len(lon_high_conf) == 0:
            continue
        if landmask_toggle == True:
            landmask = landmask_icesat2(lon_high_conf,lat_high_conf,lon_coast,lat_coast,landmask_c_file,landmask_inside_flag)
            lon_high_conf = lon_high_conf[landmask]
            lat_high_conf = lat_high_conf[landmask]
            h_high_conf = h_high_conf[landmask]
            delta_time_total_high_conf = delta_time_total_high_conf[landmask]
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_masked.txt'
            if beam_flag == True:
                beam_high_conf = beam_high_conf[landmask]
        else:
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf.txt'
        utc_time_high_conf = gps2utc(delta_time_total_high_conf)
        
        if timestamp_toggle == True:
            if beam_flag == True:
                np.savetxt(icesat2_file.replace('.txt','_beam.txt'),np.c_[lon_high_conf,lat_high_conf,h_high_conf,utc_time_high_conf.astype(object),beam_high_conf.astype(object)],fmt='%.6f,%.6f,%.6f,%s,%s',delimiter=',')
            else:
                np.savetxt(icesat2_file,np.c_[lon_high_conf,lat_high_conf,h_high_conf,utc_time_high_conf.astype(object)],fmt='%.6f,%.6f,%.6f,%s',delimiter=',')
        else:
            if beam_flag == True:
                np.savetxt(icesat2_file.replace('.txt','_beam.txt'),np.c_[lon_high_conf,lat_high_conf,h_high_conf,beam_high_conf.astype(object)],fmt='%.6f,%.6f,%.6f,%s',delimiter=',')
            else:
                np.savetxt(icesat2_file,np.c_[lon_high_conf,lat_high_conf,h_high_conf],fmt='%.6f,%.6f,%.6f',delimiter=',')
        
        if SRTM_toggle:
            SRTM_cond = SRTM_filter_icesat2(lon_high_conf,lat_high_conf,h_high_conf,icesat2_file,icesat2_dir,df_extents.iloc[i],user,pw,SRTM_threshold,EGM96_path)
            lon_high_conf_SRTM = lon_high_conf[SRTM_cond]
            lat_high_conf_SRTM = lat_high_conf[SRTM_cond]
            h_high_conf_SRTM = h_high_conf[SRTM_cond]
            delta_time_total_high_conf_SRTM = delta_time_total_high_conf[SRTM_cond]
            utc_time_high_conf_SRTM = gps2utc(delta_time_total_high_conf_SRTM)
            if beam_flag == True:
                beam_high_conf_SRTM = beam_high_conf[SRTM_cond]
            if landmask_toggle:
                icesat2_srtm_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_masked_SRTM_filtered_threshold_{SRTM_threshold_str}_m.txt'
            else:
                icesat2_srtm_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_SRTM_filtered_threshold_{SRTM_threshold_str}_m.txt'
            if timestamp_toggle:
                if beam_flag == True:
                    np.savetxt(icesat2_srtm_file.replace('.txt','_beam.txt'),np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM,utc_time_high_conf_SRTM.astype(object),beam_high_conf_SRTM.astype(object)],fmt='%.6f,%.6f,%.6f,%s,%s',delimiter=',')
                else:
                    np.savetxt(icesat2_srtm_file,np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM,utc_time_high_conf_SRTM.astype(object)],fmt='%.6f,%.6f,%.6f,%s',delimiter=',')
            else:
                if beam_flag == True:
                    np.savetxt(icesat2_srtm_file.replace('.txt','_beam.txt'),np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM,beam_high_conf_SRTM.astype(object)],fmt='%.6f,%.6f,%.6f,%s',delimiter=',')
                else:
                    np.savetxt(icesat2_srtm_file,np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM],fmt='%.6f,%.6f,%.6f',delimiter=',')
        print(f'Done with {city_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(' ')

if __name__ == '__main__':
    main()
