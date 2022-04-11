import os, sys
import glob
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon, MultiPoint
import datetime
from datetime import date
from icesat2_functions import strip_gdalinfo_lonlat,cat_str_API,landmask,gps2utc,inpoly
from atl12_cat_str import atl12_cat_str
import getpass
from osgeo import gdal, gdalconst

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###

#This script will download ICESat-2 ATL12 ocean heights for a given region.
#This script assumes data is collected over water and apply time-variable geophysical corrections.
#The point cloud will be masked with a given shapefile (e.g. a coastline).
#Output is a .txt file with ICESat-2 ATL12 data in the format:
#Longitude [deg], Latitude [deg], Height [m above WGS84], Time [UTC]


################
##Define paths##
################

input_file = '/home/eheijkoop/INPUTS/ATL12_Input.txt'
osm_shp_path = '/home/eheijkoop/ftpspace/DATA_REPOSITORY/Coast/land-polygons-complete-4326/land_polygons.shp'
tmp_dir = '/home/eheijkoop/.tmp/'
icesat2_dir = '/home/eheijkoop/ftpspace/ICESat-2/'
dtu18_path = '/home/eheijkoop/ftpspace/DATA_REPOSITORY/DTU18/DTU18MSS_WGS84_lon180.tif'

dtu18_threshold = 10 #set threshold for comparison with DTU18

#############################################
##Check if Token is still valid and load it##
#############################################

os.system('stat Token.txt > tmp_stats_token.txt')
for line in open('tmp_stats_token.txt'):
    if "Modify:" in line:
        date_str = line

date_str = date_str.split(" ")
date_str = date_str[1]

today_datetime = date.today()
token_datetime = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
delta_datetime = today_datetime - token_datetime
if delta_datetime.days > 30:
    print('WARNING!')
    print('Token expired!')
    print('Exiting...')
    sys.exit()

for line in open('Token.txt'):
    token=line
token = token.rstrip('\n')

#"header=None" if you don't want a header in your input file
#"heard=0" if you do want a header, like "Name,lon_min,lon_max,lat_min,lat_max"
df_extents = pd.read_csv(input_file,header=0,names=['city','lon_min','lon_max','lat_min','lat_max'],dtype={'city':'str','lon_min':'float','lon_max':'float','lat_min':'float','lat_max':'float'})

for i in range(len(df_extents.city)):
    ##############
    ##FIND DTU18##
    ##############
    '''
    #DTU18 is sampled at the mean of lon/lat extents, to find local MSL
    f_dtu18 = open(tmp_dir + 'tmp_lonlat_out.txt','w')
    np.savetxt(f_dtu18,np.c_[np.mean([df_extents.lon_min[i],df_extents.lon_max[i]]),np.mean([df_extents.lat_min[i],df_extents.lat_max[i] ] ),0 ],fmt='%10.5f',delimiter=',')
    f_dtu18.close()
    os.system('gmt grdtrack ' + tmp_dir + 'tmp_lonlat_out.txt -G' + dtu18_path + ' > ' + tmp_dir + 'tmp_dtu18_sample.txt')

    df_dtu18 = pd.read_csv(tmp_dir + 'tmp_dtu18_sample.txt',header=None,names=['lon','lat','throwaway','dtu18'],dtype={'lon':'float','lat':'float','throwaway':'float','dtu18':'float'})
    dtu18_msl = df_dtu18.dtu18[0]
    os.system('rm ' + tmp_dir + 'tmp_dtu18_sample.txt')
    os.system('rm ' + tmp_dir + 'tmp_lonlat_out.txt')
    '''
    ##############
    ##SUBSET OSM##
    ##############
    
    city_name = df_extents.city[i]
    if not os.path.isdir(icesat2_dir+city_name):
        os.system('mkdir ' + icesat2_dir + city_name)
    
    lon_min_str = str(df_extents.lon_min[i])
    lon_max_str = str(df_extents.lon_max[i])
    lat_min_str = str(df_extents.lat_min[i])
    lat_max_str = str(df_extents.lat_max[i])

    '''
    lon_min_str_osm = str(df_extents.lon_min[i]-0.1)
    lon_max_str_osm = str(df_extents.lon_max[i]+0.1)
    lat_min_str_osm = str(df_extents.lat_min[i]-0.1)
    lat_max_str_osm = str(df_extents.lat_max[i]+0.1)


    extents_str = lon_min_str_osm + ' ' + lat_min_str_osm + ' ' + lon_max_str_osm + ' ' + lat_max_str_osm
    output_shp = icesat2_dir + city_name + '/' + city_name + '.shp'
    subset_shp_path = output_shp
    shp_command = 'ogr2ogr ' + output_shp + ' ' +  osm_shp_path + ' -clipsrc ' + extents_str
    os.system(shp_command)

    full_msl_file = icesat2_dir+city_name+'/'+city_name+'_ocean_heights.txt'
    full_msl_time_file = icesat2_dir+city_name+'/'+city_name+'_ocean_heights_time.txt'
    full_msl_file_dtu18_check = icesat2_dir+city_name+'/'+city_name+'_ocean_heights_DTU18_filter.txt'
    full_msl_time_file_dtu18_check = icesat2_dir+city_name+'/'+city_name+'_ocean_heights_DTU18_filter_time.txt'
    '''
    #####################
    ##Download ICESat-2##
    #####################
    #See: https://nsidc.org/support/how/how-do-i-programmatically-request-data-services#curl

    token_command = 'token='+token
    site_command = 'https://n5eil02u.ecs.nsidc.org/egi/request?'
    email_command = 'email=false'
    short_name = 'ATL12'
    coverage_command = 'coverage='
    beam_list = ['1l','1r','2l','2r','3l','3r']
    for beam in beam_list:
        coverage_command = coverage_command + atl12_cat_str(beam)
    coverage_command = coverage_command + '/orbit_info/sc_orient,/ancillary_data/atlas_sdp_gps_epoch,/ancillary_data/data_start_utc,/ancillary_data/data_end_utc'

    #May need to periodically update ICESat-2 version number!
    short_name_command = 'short_name=' + short_name + '&version=003'
    time_command = 'time=2019-01-01T00:00:00,2019-12-31T23:59:59&'

    bounding_box_command = 'bounding_box='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    bbox_command = 'bbox='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    shape_command = bounding_box_command + '&' + bbox_command + '&'

    #API will give at most 10 subsetted .H5 files in a single zip file.
    #Iterating the command page_num=N (where N is page number) allows you to get everything
    #API doesn't say how many files are available, so must check response-header.txt and evaluate code
      #200 is good, download data
      #501 means it's empty
      #404 unknown is a common error in URLs, these numbers are in the same category
    page_number = 1
    #iterate over page numbers
    page_condition = True

    while page_condition:
        page_command = 'page_num='+str(page_number)
        full_command = 'curl -O -J -k --dump-header response-header.txt \"' + site_command + '&' + short_name_command + '&' + token_command + '&' + email_command + '&' + shape_command + time_command + coverage_command + '&' + page_command + '\"'
        
        os.system('echo ' + full_command + ' > tmp_command.txt')

        #print('Running this command:')
        #print(full_command)
        os.system(full_command)

        with open('response-header.txt','r') as f2:
            response_line = f2.readline().replace('\n','')
        if response_line[9:12] == '200':
            page_number = page_number + 1
        elif response_line[9:12] == '501':
            page_condition = False
        else:
            print('Something bad happened.')
            print('Exiting...')
            page_condition = False
    
    #####################
    ##Unzip & Move Data##
    #####################

    os.system('mv *zip ' + icesat2_dir+city_name + '/')
    os.system('mv *h5 ' + icesat2_dir+city_name + '/')
    os.system('rm *xml')
    os.system('mv response-header.txt response-header_'+city_name+'_'+str(page_number)+'.txt')

    os.system('unzip \'' + icesat2_dir+city_name + '/*zip\' -d ' + icesat2_dir+city_name + '/')
    os.system('mv ' + icesat2_dir+city_name + '/*/processed*.h5 ' + icesat2_dir+city_name+ '/')
    #os.system('rm -rf ' + icesat2_dir+city_name + '/1*')
    #os.system('rm ' + icesat2_dir+city_name + '/5*zip')
    os.system('find ' + icesat2_dir+city_name + '/*h5 -printf "%f\\'+'n" > ' + icesat2_dir+city_name + '/icesat2_list.txt')
    
    icesat2_list = icesat2_dir+city_name + '/icesat2_list.txt'
    
    ############
    ##Analysis##
    ############

    #shp_data = gpd.read_file(subset_shp_path)
    #crs = {'init': 'epsg:4326'}

    with open(icesat2_list) as f3:
        file_list = f3.read().splitlines()

    beam_list_r = ['gt1r','gt2r','gt3r']
    beam_list_l = ['gt1l','gt2l','gt3l']

    #Initialize arrays and start reading .h5 files
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    h = np.empty([0,1],dtype=float)
    signal_conf = np.empty(shape=[0,5],dtype=float)

    lon_high_med_conf = np.empty([0,1],dtype=float)
    lat_high_med_conf = np.empty([0,1],dtype=float)
    h_high_med_conf = np.empty([0,1],dtype=float)
    
    ocean_tide_high_med_conf = np.empty([0,1],dtype=float)
    dac_high_med_conf = np.empty([0,1],dtype=float)
    delta_time_total_high_med_conf = np.empty([0,1],dtype=float)





    for h5_file in file_list:

        full_file = icesat2_dir + city_name + '/' + h5_file
        atl03_file = h5py.File(full_file,'r')
        list(atl03_file.keys())

        sc_orient = atl03_file['/orbit_info/sc_orient']
        sc_orient = sc_orient[0]

        if sc_orient == 1:
            beam_list_req = beam_list_r
        elif sc_orient == 0:
            beam_list_req = beam_list_l
        elif sc_orient == 2:
            continue



        for beam in beam_list_req:
            #Some beams don't actually have any height data in them, so this is done to skip those
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

            tmp_delta_time_geophys_corr = np.asarray(atl03_file['/'+beam+'/geophys_corr/delta_time']).squeeze()
            #Don't really need time of geophysical corrections here
            #tmp_delta_time_total_geophys_corr = tmp_sdp + tmp_delta_time_geophys_corr

            tmp_ocean_tide = np.asarray(atl03_file['/'+beam+'/geophys_corr/tide_ocean']).squeeze()
            tmp_dac = np.asarray(atl03_file['/'+beam+'/geophys_corr/dac']).squeeze()

            flag_ocean_tide_dac = np.logical_or(tmp_ocean_tide>1e20,tmp_dac>1e20)
            
            tmp_delta_time_geophys_corr = tmp_delta_time_geophys_corr[np.invert(flag_ocean_tide_dac)]
            #tmp_delta_time_total_geophys_corr = tmp_delta_time_total_geophys_corr[np.invert(flag_ocean_tide_dac)]
            tmp_ocean_tide = tmp_ocean_tide[np.invert(flag_ocean_tide_dac)]
            tmp_dac = tmp_dac[np.invert(flag_ocean_tide_dac)]


            tmp_signal_conf = np.asarray(atl03_file['/'+beam+'/heights/signal_conf_ph'])

            tmp_high_med_conf = np.logical_or(tmp_signal_conf[:,1]==3,tmp_signal_conf[:,1]==4)
            tmp_lon_high_med_conf = tmp_lon[tmp_high_med_conf]
            tmp_lat_high_med_conf = tmp_lat[tmp_high_med_conf]
            tmp_h_high_med_conf = tmp_h[tmp_high_med_conf]
            tmp_delta_time_high_med_conf = tmp_delta_time[tmp_high_med_conf]
            tmp_delta_time_total_high_med_conf = tmp_delta_time_total[tmp_high_med_conf]
            
            if np.logical_or(tmp_ocean_tide.size==0,tmp_dac.size==0):
                print('No Ocean Tide or DAC, skipping this beam!')
                continue

            tmp_ocean_tide = np.interp(tmp_delta_time_high_med_conf,tmp_delta_time_geophys_corr,tmp_ocean_tide)
            tmp_dac = np.interp(tmp_delta_time_high_med_conf,tmp_delta_time_geophys_corr,tmp_dac)
            
            
            ##########
            ##Append##
            ##########

            lon_high_med_conf = np.append(lon_high_med_conf,tmp_lon_high_med_conf)
            lat_high_med_conf = np.append(lat_high_med_conf,tmp_lat_high_med_conf)
            h_high_med_conf = np.append(h_high_med_conf,tmp_h_high_med_conf)
            delta_time_total_high_med_conf = np.append(delta_time_total_high_med_conf,tmp_delta_time_total_high_med_conf)
            ocean_tide_high_med_conf = np.append(ocean_tide_high_med_conf,tmp_ocean_tide)
            dac_high_med_conf = np.append(dac_high_med_conf,tmp_dac)
            
    
    ############
    ##Landmask##
    ############
    
    print('Running landmask...')
    t_start = datetime.datetime.now()
    #create boolean array of ICESat-2 returns, whether or not they're inside the given shapefile 
    landmask = inpoly(lon_high_med_conf,lat_high_med_conf,subset_shp_path,True)
    t_end = datetime.datetime.now()
    print('Landmask done.')

    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    dt_hour, dt_min = divmod(dt_min,60)
    print('It took:')
    print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
    

    #"landmask==False" because we want data outside the shapefile (i.e. water)
    lon_high_med_conf_masked = lon_high_med_conf[landmask==False]
    lat_high_med_conf_masked = lat_high_med_conf[landmask==False]
    h_high_med_conf_masked = h_high_med_conf[landmask==False]
    delta_time_total_high_med_conf_masked = delta_time_total_high_med_conf[landmask==False]
    ocean_tide_high_med_conf_masked = ocean_tide_high_med_conf[landmask==False]
    dac_high_med_conf_masked = dac_high_med_conf[landmask==False]
    
    utc_time_high_med_conf_masked = gps2utc(delta_time_total_high_med_conf_masked)

    h_corrected_high_med_conf_masked = h_high_med_conf_masked - ocean_tide_high_med_conf_masked - dac_high_med_conf_masked
    
    dtu18_msl_check = np.abs(h_corrected_high_med_conf_masked - dtu18_msl) < dtu18_threshold

    lon_high_med_conf_masked_dtu18_check = lon_high_med_conf_masked[dtu18_msl_check]
    lat_high_med_conf_masked_dtu18_check = lat_high_med_conf_masked[dtu18_msl_check]
    h_corrected_high_med_conf_masked_dtu18_check = h_corrected_high_med_conf_masked[dtu18_msl_check]
    delta_time_total_high_med_conf_masked_dtu18_check = delta_time_total_high_med_conf_masked[dtu18_msl_check]
    utc_time_high_med_conf_masked_dtu18_check = gps2utc(delta_time_total_high_med_conf_masked_dtu18_check)

    #This is the mean of all ICESat-2 ocean heights (with and without checking with DTU18)
    #May want to save these too, up to you
    h_msl = np.mean(h_corrected_high_med_conf_masked)
    h_msl_dtu18_check = np.mean(h_corrected_high_med_conf_masked_dtu18_check)

    f1 = open(full_msl_file,'w')
    f1a = open(full_msl_time_file,'w')
    f10 = open(full_msl_file_dtu18_check,'w')
    f10a = open(full_msl_time_file_dtu18_check,'w')
    np.savetxt(f1,np.c_[lon_high_med_conf_masked,lat_high_med_conf_masked,h_corrected_high_med_conf_masked],fmt='%10.5f',delimiter=',')
    np.savetxt(f1a,np.c_[utc_time_high_med_conf_masked],fmt='%s')
    np.savetxt(f10,np.c_[lon_high_med_conf_masked_dtu18_check,lat_high_med_conf_masked_dtu18_check,h_corrected_high_med_conf_masked_dtu18_check],fmt='%10.5f',delimiter=',')
    np.savetxt(f10a,np.c_[utc_time_high_med_conf_masked_dtu18_check],fmt='%s')

    
    f1.close()
    f1a.close()
    f10.close()
    f10a.close()


    os.system('paste -d , '+full_msl_file+' '+full_msl_time_file+ ' > ' + tmp_dir + 'tmp_paste.txt')
    os.system('mv ' + tmp_dir + 'tmp_paste.txt ' + full_msl_file)
    os.system('rm ' + full_msl_time_file)

    os.system('paste -d , '+full_msl_file_dtu18_check+' '+full_msl_time_file_dtu18_check+ ' > ' + tmp_dir + 'tmp_paste.txt')
    os.system('mv ' + tmp_dir + 'tmp_paste.txt ' + full_msl_file_dtu18_check)
    os.system('rm ' + full_msl_time_file_dtu18_check)
    
    now = datetime.datetime.now()
    print('Done with ' + city_name + ' at:')
    print(now.strftime("%Y-%m-%d %H:%M:%S"))




