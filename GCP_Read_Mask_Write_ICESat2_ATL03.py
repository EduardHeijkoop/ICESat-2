import os
import numpy as np
import pandas as pd
import datetime
import getpass
import configparser
import argparse
import warnings
import subprocess
import sys
import glob

from icesat2_utils import get_osm_extents,create_bbox,move_icesat2,download_icesat2,check_h5_count,check_password_nasa_earthdata
from icesat2_utils import gps2utc,parallel_landmask,delta_time_to_orientation,beam_orientation_to_strength
from gcp_utils import analyze_icesat2_land, copernicus_filter_icesat2

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
    parser.add_argument('--machine',default='t',help='Machine to run on.',choices=['t','b','local'])
    parser.add_argument('--landmask',action='store_true',default=False,help='Toggle to mask photons over land/water.')
    parser.add_argument('--time',action='store_true',default=False,help='Toggle to print timestamps.')
    parser.add_argument('--beams',action='store_true',default=False,help='Toggle to print beams.')
    parser.add_argument('--strength',default='strong',type=str,help='Which beams to analyze.',choices=['strong','weak','all'])
    parser.add_argument('--weight',action='store_true',default=False,help='Toggle to incorporate weight parameter.')
    parser.add_argument('--sigma',action='store_true',default=False,help='Toggle to print sigma.')
    parser.add_argument('--fpb',action='store_true',default=False,help='Toggle to incorporate first photon bias.')
    parser.add_argument('--N_cpus',default=1,type=int,help='Number of CPUs to use.')
    parser.add_argument('--version',default=6,type=int,help='Which version to download.')
    parser.add_argument('--copernicus',action='store_true',default=False,help='Toggle to filter with Copernicus DEM.')
    parser.add_argument('--keep_files',action='store_true',default=False,help='Toggle to keep Copernicus DEM files.')
    args = parser.parse_args()
    machine_name = args.machine
    landmask_flag = args.landmask
    timestamp_flag = args.time
    beam_flag = args.beams
    beam_strength = args.strength
    weight_flag = args.weight
    sigma_flag = args.sigma
    fpb_flag = args.fpb
    N_cpus = args.N_cpus
    version = args.version
    copernicus_flag = args.copernicus
    keep_files_flag = args.keep_files

    if beam_strength == 'all':
        #Must have beams to distinguish between strong and weak
        beam_flag = True

    input_file = config.get('GCP_PATHS','input_file') #Input file with location name,lon_min,lon_max,lat_min,lat_max (1 header line)
    osm_shp_file = config.get('GENERAL_PATHS','osm_shp_file') #OpenStreetMap land polygons, available at https://osmdata.openstreetmap.de/data/land-polygons.html (use WGS84, not split)
    icesat2_dir = config.get('GCP_PATHS','icesat2_dir') #output directory, which will be populated by subdirectories named after your input
    error_log_file = config.get('GCP_PATHS','error_log_file') #file to write errors to
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file') #file with C function pnpoly, "point in polygon", to perform landmask
    landmask_inside_flag = config.getint('GCP_CONSTANTS','landmask_inside_flag') #flag to find points inside (1) or outside (0) polygon

    user = config.get('GENERAL_CONSTANTS','earthdata_username') #Your NASA EarthData username
    pw = getpass.getpass('NASA EarthData password:') #Your NASA EarthData password
    pw_check = check_password_nasa_earthdata(user,pw)
       
    if copernicus_flag:
        copernicus_threshold = config.getfloat('GCP_CONSTANTS','Copernicus_Threshold') #set your copernicus threshold here
        copernicus_threshold_str = str(copernicus_threshold).replace('.','p') #replace decimal point with p for file name
        EGM2008_path = config.get('GCP_PATHS','EGM2008_path') #supplied on github

    if machine_name == 'b':
        osm_shp_file = osm_shp_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        if copernicus_flag:
            EGM2008_path = EGM2008_path.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/') 
    elif machine_name == 'local':
        osm_shp_file = osm_shp_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        icesat2_dir = icesat2_dir.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/','/media/heijkoop/DATA/')
        error_log_file = error_log_file.replace('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/','/media/heijkoop/DATA/')
        landmask_c_file = landmask_c_file.replace('/home/eheijkoop/Scripts/','/media/heijkoop/DATA/Dropbox/TU/PhD/Github/')
        if copernicus_flag:
            EGM2008_path = EGM2008_path.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/GEOID/')
            
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
        lon_coast,lat_coast,shp_data = get_osm_extents(df_extents.iloc[i],osm_shp_file,icesat2_dir)
        bbox_code = create_bbox(icesat2_dir,df_extents.iloc[i])
        if bbox_code is not None:
            continue
        N_h5 = check_h5_count(f'{icesat2_dir}{city_name}')
        if N_h5 > 0:
            h5_check = input('HDF5 files already found in directory! Analyze these instead? y/n\n'\
                             'y: analyze current HDF5 files (lon/lat extents may not match current input).\n'\
                             'n: remove current HDF5 files, download new set and analyze those.\n')
            input_option_list = ['y','yes']
            if h5_check.lower() not in input_option_list:
                subprocess.run(f'rm {icesat2_dir}{city_name}/*.h5',shell=True)
                sync_async_code = download_icesat2(user,pw,df_extents.iloc[i],version)
                if sync_async_code is None:
                    continue
                move_code = move_icesat2(icesat2_dir,df_extents.iloc[i])
                if move_code is not None:
                    continue
        else:
            sync_async_code = download_icesat2(user,pw,df_extents.iloc[i],version)
            if sync_async_code is None:
                continue
            move_code = move_icesat2(icesat2_dir,df_extents.iloc[i])
            if move_code is not None:
                continue
        lon_high_conf,lat_high_conf,h_high_conf,delta_time_total_high_conf,beam_high_conf,sigma_high_conf = analyze_icesat2_land(icesat2_dir,city_name,shp_data,beam_flag,beam_strength,sigma_flag,weight_flag,fpb_flag)
        if beam_strength == 'all':
            sc_orient = delta_time_to_orientation(delta_time_total_high_conf)
            strength_high_conf = beam_orientation_to_strength(beam_high_conf,sc_orient)
        if len(lon_high_conf) == 0:
            continue
        if landmask_flag == True:
            landmask = parallel_landmask(lon_high_conf,lat_high_conf,lon_coast,lat_coast,landmask_c_file,landmask_inside_flag,N_cpus=N_cpus)
            lon_high_conf = lon_high_conf[landmask]
            lat_high_conf = lat_high_conf[landmask]
            h_high_conf = h_high_conf[landmask]
            delta_time_total_high_conf = delta_time_total_high_conf[landmask]
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_masked.txt'
            if beam_flag == True:
                beam_high_conf = beam_high_conf[landmask]
            if sigma_flag == True:
                sigma_high_conf = sigma_high_conf[landmask]
            if beam_strength == 'all':
                strength_high_conf = strength_high_conf[landmask]
        else:
            icesat2_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf.txt'
        utc_time_high_conf = gps2utc(delta_time_total_high_conf)
        
        if beam_strength == 'weak':
            icesat2_file = icesat2_file.replace('_high_conf','_high_conf_weak')
        elif beam_strength == 'all':
            icesat2_file = icesat2_file.replace('_high_conf','_high_conf_all_beams')
        file_list = [icesat2_file]
        np.savetxt(icesat2_file,np.c_[lon_high_conf,lat_high_conf,h_high_conf],fmt='%.6f,%.6f,%.6f',delimiter=',',header='lon,lat,height_icesat2',comments='')
        if timestamp_flag == True:
            icesat2_time_file = icesat2_file.replace('.txt','_time.txt')
            file_list.append(icesat2_time_file)
            np.savetxt(icesat2_time_file,utc_time_high_conf.astype(object),fmt='%s',delimiter=',',header='time',comments='')
        if beam_flag == True:
            icesat2_beam_file = icesat2_file.replace('.txt','_beam.txt')
            file_list.append(icesat2_beam_file)
            np.savetxt(icesat2_beam_file,beam_high_conf.astype(object),fmt='%s',delimiter=',',header='beam',comments='')
        if sigma_flag == True:
            icesat2_sigma_file = icesat2_file.replace('.txt','_sigma.txt')
            file_list.append(icesat2_sigma_file)
            np.savetxt(icesat2_sigma_file,sigma_high_conf,fmt='%.6f',delimiter=',',header='sigma',comments='')
        if beam_strength == 'all':
            icesat2_strength_file = icesat2_file.replace('.txt','_strength.txt')
            file_list.append(icesat2_strength_file)
            np.savetxt(icesat2_strength_file,strength_high_conf.astype(object),fmt='%s',delimiter=',',header='strength',comments='')

        if len(file_list) > 1:
            tmp_file = icesat2_file.replace('.txt','_tmp.txt')
            paste_command = f'paste -d , {" ".join(file_list)} > {tmp_file}'
            move_command = f'mv {tmp_file} {icesat2_file}'
            rm_command = f'rm {" ".join(file_list[1:])}'
            subprocess.run(paste_command,shell=True)
            subprocess.run(move_command,shell=True)
            subprocess.run(rm_command,shell=True)
        
        if copernicus_flag == True:
            copernicus_cond = copernicus_filter_icesat2(lon_high_conf,lat_high_conf,icesat2_file,icesat2_dir,city_name,copernicus_threshold,EGM2008_path,keep_files_flag=keep_files_flag)
            lon_high_conf_copernicus = lon_high_conf[copernicus_cond]
            lat_high_conf_copernicus = lat_high_conf[copernicus_cond]
            h_high_conf_copernicus = h_high_conf[copernicus_cond]
            delta_time_total_high_conf_copernicus = delta_time_total_high_conf[copernicus_cond]
            if landmask_flag:
                icesat2_copernicus_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_masked_copernicus_filtered_threshold_{copernicus_threshold_str}_m.txt'
            else:
                icesat2_copernicus_file = f'{icesat2_dir}{city_name}/{city_name}_ATL03_high_conf_copernicus_filtered_threshold_{copernicus_threshold_str}_m.txt'
            if beam_strength == 'weak':
                icesat2_copernicus_file = icesat2_copernicus_file.replace('_high_conf','_high_conf_weak')
            elif beam_strength == 'all':
                icesat2_copernicus_file = icesat2_copernicus_file.replace('_high_conf','_high_conf_all_beams')
            file_list_copernicus = [icesat2_copernicus_file]
            np.savetxt(icesat2_copernicus_file,np.c_[lon_high_conf_copernicus,lat_high_conf_copernicus,h_high_conf_copernicus],fmt='%.6f,%.6f,%.6f',delimiter=',',header='lon,lat,height_icesat2',comments='')
            if timestamp_flag == True:
                utc_time_high_conf_copernicus = gps2utc(delta_time_total_high_conf_copernicus)
                icesat2_copernicus_time_file = icesat2_copernicus_file.replace('.txt','_time.txt')
                file_list_copernicus.append(icesat2_copernicus_time_file)
                np.savetxt(icesat2_copernicus_time_file,utc_time_high_conf_copernicus.astype(object),fmt='%s',delimiter=',',header='time',comments='')
            if beam_flag == True:
                beam_high_conf_copernicus = beam_high_conf[copernicus_cond]
                icesat2_copernicus_beam_file = icesat2_copernicus_file.replace('.txt','_beam.txt')
                file_list_copernicus.append(icesat2_copernicus_beam_file)
                np.savetxt(icesat2_copernicus_beam_file,beam_high_conf_copernicus.astype(object),fmt='%s',delimiter=',',header='beam',comments='')
            if sigma_flag == True:
                sigma_high_conf_copernicus = sigma_high_conf[copernicus_cond]
                icesat2_copernicus_sigma_file = icesat2_copernicus_file.replace('.txt','_sigma.txt')
                file_list_copernicus.append(icesat2_copernicus_sigma_file)
                np.savetxt(icesat2_copernicus_sigma_file,sigma_high_conf_copernicus,fmt='%.6f',delimiter=',',header='sigma',comments='')
            if beam_strength == 'all':
                strength_high_conf_copernicus = strength_high_conf[copernicus_cond]
                icesat2_copernicus_strength_file = icesat2_copernicus_file.replace('.txt','_strength.txt')
                file_list_copernicus.append(icesat2_copernicus_strength_file)
                np.savetxt(icesat2_copernicus_strength_file,strength_high_conf_copernicus.astype(object),fmt='s',delimiter=',',header='strength',comments='')

            if len(file_list_copernicus) > 1:
                tmp_file = icesat2_copernicus_file.replace('.txt','_tmp.txt')
                paste_command = f'paste -d , {" ".join(file_list_copernicus)} > {tmp_file}'
                move_command = f'mv {tmp_file} {icesat2_copernicus_file}'
                rm_command = f'rm {" ".join(file_list_copernicus[1:])}'
                subprocess.run(paste_command,shell=True)
                subprocess.run(move_command,shell=True)
                subprocess.run(rm_command,shell=True)

        print(f'Done with {city_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(' ')

if __name__ == '__main__':
    main()
