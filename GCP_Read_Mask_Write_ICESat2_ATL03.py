import os
import glob
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import ctypes as c
from osgeo import gdal, gdalconst
import datetime
import subprocess
import getpass
import socket
import xml.etree.ElementTree as ET
import shapely

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###
#Update March 2021: now does landmasking in C
#Update December 2021: rewrite into functions

#This script will download ICESat-2 ATL03 geolocated photons for a given region.
#The point cloud will be masked with a given shapefile (e.g. a coastline), originally used as ground control points (GCPs)
#Output is a .txt file with ICESat-2 ATL03 data in the format:
#Longitude [deg], Latitude [deg], Height [m above WGS84], Time [UTC]

def get_lonlat_shp(shp_path):
    #Given a shapefile (.shp), returns longitude and latitude arrays
    #of all individual polygons, separated by NaNs
    #Polygons within polygons will be included here
    shp = gpd.read_file(shp_path)
    lon_coast = np.empty([0,1],dtype=float)
    lat_coast = np.empty([0,1],dtype=float)
    for ii in range(len(shp)):
        tmp_geom = shp.geometry[ii]
        tmp_geom_type = shp.geometry[ii].geom_type
        if tmp_geom_type == 'Polygon':
            tmp = np.asarray(shp.geometry[ii].exterior.xy)
            lon_coast = np.append(lon_coast,tmp[0,:])
            lon_coast = np.append(lon_coast,np.nan)
            lat_coast = np.append(lat_coast,tmp[1,:])
            lat_coast = np.append(lat_coast,np.nan)
            if len(tmp_geom.interiors) > 0:
                for interior in tmp_geom.interiors:
                    tmp_int = np.asarray(interior.coords.xy)
                    lon_coast = np.append(lon_coast,tmp_int[0,:])
                    lon_coast = np.append(lon_coast,np.nan)
                    lat_coast = np.append(lat_coast,tmp_int[1,:])
                    lat_coast = np.append(lat_coast,np.nan)
        elif tmp_geom_type == 'MultiPolygon':
            tmp_list = list(shp.boundary[ii])
            for jj in range(len(tmp_list)):
                tmp = np.asarray(tmp_list[jj].coords.xy)
                lon_coast = np.append(lon_coast,tmp[0,:])
                lon_coast = np.append(lon_coast,np.nan)
                lat_coast = np.append(lat_coast,tmp[1,:])
                lat_coast = np.append(lat_coast,np.nan)
        elif tmp_geom_type == 'LineString':
            tmp = np.asarray(shp.geometry[ii].xy)
            lon_coast = np.append(lon_coast,tmp[0,:])
            lon_coast = np.append(lon_coast,np.nan)
            lat_coast = np.append(lat_coast,tmp[1,:])
            lat_coast = np.append(lat_coast,np.nan)
    return lon_coast, lat_coast

def cat_str_API(beam):
    #Strings together hdf5 path for:
    #   photon h/lon/lat/signal_conf/sigma_h
    #   reference photon lon/lat
    #   ocean tide/DAC
    #   (geophysical corrections') delta_time
    #For all 3 strong beams
	beam_command = '/gt'+beam+'/heights/h_ph,/gt'+beam+'/heights/lon_ph,/gt'+beam+'/heights/lat_ph,/gt'+beam+'/heights/delta_time,' \
		'/gt'+beam+'/heights/signal_conf_ph,/gt'+beam+'/geolocation/sigma_h,' \
		'/gt'+beam+'/geolocation/reference_photon_lon,/gt'+beam+'/geolocation/reference_photon_lat,/gt'+beam+'/geophys_corr/tide_ocean,/gt'+beam+'/geophys_corr/dac,' \
		'/gt'+beam+'/geolocation/delta_time,/gt'+beam+'/geophys_corr/delta_time,'
	return beam_command

def gps2utc(gps_time):
    #Converts GPS time that ICESat-2 references to UTC
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = [str(x) for x in utc_time]
    return utc_time_str

def get_token(user):
    #Given username and password, get NSIDC token to download ICESat-2
    #Checks age of token file as it expires after 30 days
    #If token not present, triggers next if statement immediately
    today_datetime = datetime.datetime.now()
    if os.path.isfile('Token.xml'):
        stats_token = os.stat('Token.xml')
        token_datetime = datetime.datetime.fromtimestamp(stats_token.st_mtime)
    else:
        token_datetime = datetime.datetime.strptime('1900-01-01','%Y-%m-%d')
    token_age = today_datetime - token_datetime
    if token_age.days >= 30:
        print('Token expired or not present!')
        print('Please enter NASA EarthData password to generate new token:')
        pw = getpass.getpass()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        token_command = 'curl -X POST --header \"Content-Type: application/xml\" -d \'<token><username>'+user+'</username><password>'+pw+'</password><client_id>NSIDC_client_id</client_id><user_ip_address>'+ip_address+'</user_ip_address> </token>\' https://cmr.earthdata.nasa.gov/legacy-services/rest/tokens -o Token.xml --silent'
        subprocess.run(token_command,shell=True)
    token_tree = ET.parse('Token.xml')
    token_root = token_tree.getroot()
    token = token_root[0].text
    return token

def cleanup():
    #Cleans up a number of files that linger after download
    data_end_utc_files = glob.glob('*data_end_utc*')
    for data_end_file in data_end_utc_files:
        data_end_file = data_end_file.replace('&','\&')
        data_end_file = data_end_file.replace('=','\=')
        subprocess.run('rm '+data_end_file,shell=True)
    if os.path.exists('response-header.txt'):
        subprocess.run('rm response-header.txt',shell=True)
    if os.path.exists('error.xml'):
        subprocess.run('rm error.xml',shell=True)
    return None
    
def get_osm_extents(df_city,osm_shp_path,icesat2_dir):
    #Given lon/lat extents in a Pandas DataFrame (df_city),
    #subsets OpenStreetMap land polygons to those extents as a new shapefile
    #and returns lon/lat of that new shapefile 
    city_name = df_city.city
    lon_min_str = str(df_city.lon_min)
    lon_max_str = str(df_city.lon_max)
    lat_min_str = str(df_city.lat_min)
    lat_max_str = str(df_city.lat_max)
    extents_str = lon_min_str + ' ' + lat_min_str + ' ' + lon_max_str + ' ' + lat_max_str
    subset_shp = icesat2_dir + city_name + '/' + city_name + '.shp'
    shp_command = 'ogr2ogr ' + subset_shp + ' ' +  osm_shp_path + ' -clipsrc ' + extents_str
    subprocess.run(shp_command,shell=True)
    shp_data = gpd.read_file(subset_shp)
    lon_coast,lat_coast = get_lonlat_shp(subset_shp)
    return lon_coast,lat_coast,shp_data

def validate_date(date_text):
    date_text = str(date_text)
    try:
        date_text = datetime.datetime.strptime(date_text,'%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        date_text = date_text
    try:
        if date_text != datetime.datetime.strptime(date_text, "%Y-%m-%d").strftime('%Y-%m-%d'):
            raise ValueError
        return True
    except ValueError:
        return False

def create_bbox(icesat2_dir,df_city):
    city_name = df_city.city
    lon_min = df_city.lon_min
    lon_max = df_city.lon_max
    lat_min = df_city.lat_min
    lat_max = df_city.lat_max

    ll = [lon_min,lat_min]
    ul = [lon_min,lat_max]
    ur = [lon_max,lat_max]
    lr = [lon_max,lat_min]

    bbox_geom = shapely.geometry.Polygon([ll,ul,ur,lr,ll])
    gdf_bbox = gpd.GeoDataFrame(pd.DataFrame({'Bbox':[city_name]}),geometry=[bbox_geom],crs='EPSG:4326')
    gdf_bbox.to_file(icesat2_dir + city_name + '/' + city_name + '_bbox.shp')
    return None

def download_icesat2(df_city,token,error_log_file):
    #Given lon/lat extents in a Pandas DataFrame (df_city),
    #downloads ICESat-2 ATL03 version 5 geolocated photons
    city_name = df_city.city
    t_start = df_city.t_start
    t_end = df_city.t_end
    t_start_valid = validate_date(t_start)
    t_end_valid = validate_date(t_end)
    lon_min_str = str(df_city.lon_min)
    lon_max_str = str(df_city.lon_max)
    lat_min_str = str(df_city.lat_min)
    lat_max_str = str(df_city.lat_max)
    token_command = 'token='+token
    site_command = 'https://n5eil02u.ecs.nsidc.org/egi/request?'
    email_command = 'email=false'
    short_name = 'ATL03'
    coverage_command = 'coverage='
    beam_list = ['1l','1r','2l','2r','3l','3r']
    for beam in beam_list:
        coverage_command = coverage_command + cat_str_API(beam)
    coverage_command = coverage_command + '/orbit_info/sc_orient,/ancillary_data/atlas_sdp_gps_epoch,/ancillary_data/data_start_utc,/ancillary_data/data_end_utc'
    short_name_command = 'short_name=' + short_name + '&version=005'
    if t_start_valid == True:
        t_start = datetime.datetime.strptime(t_start,'%Y-%m-%d').strftime('%Y-%m-%d')
    else:
        t_start = '2018-10-01'

    if t_end_valid == True:
        t_end = datetime.datetime.strptime(t_end,'%Y-%m-%d').strftime('%Y-%m-%d')
    else:
        t_end = datetime.datetime.now().strftime('%Y-%m-%d')

    if np.logical_and(t_start_valid==False,t_end_valid==False):
        time_command = ''
    else:
        time_command = 'time='+t_start+'T00:00:00,'+t_end+'T23:59:59&'
    bounding_box_command = 'bounding_box='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    bbox_command = 'bbox='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    shape_command = bounding_box_command + '&' + bbox_command + '&'
    page_number = 1
    page_condition = True
    while page_condition:
        page_command = 'page_num='+str(page_number)
        full_command = 'curl -O -J -k --dump-header response-header.txt \"' + \
            site_command + '&' + short_name_command + '&' + token_command + '&' + \
            email_command + '&' + shape_command + time_command + \
            coverage_command + '&' + page_command + '\"'
        subprocess.run(full_command,shell=True)
        with open('response-header.txt','r') as f2:
            response_line = f2.readline().replace('\n','')
        if response_line[9:12] == '200':
            page_number = page_number + 1
        elif response_line[9:12] == '204':
            print('End of download.')
            page_condition = False
        elif response_line[9:12] == '501':
            page_condition = False
        else:
            print('Something bad happened.')
            print('Exiting...')
            page_condition = False
    if page_number == 1:
        print('Nothing was downloaded.')
        print('Check extents - possibly no coverage!')
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(error_log_file,'a') as text_file:
            text_file.write(now_str + ': ' + city_name + ' - No data download.\n')
            print('No data downloaded!')
        return 0
    else:
        return None

def move_icesat2(icesat2_dir,df_city):
    #move .zip files that are downloaded by the download_icesat2 function to the correct directory
    #unzip them there and put all .h5 files in that directory
    #remove empty subdirectories that linger after unzipping
    #remove zip files
    #create icesat2_list.txt file with list of .h5 files 
    city_name = df_city.city
    subprocess.run('mv *zip ' + icesat2_dir+city_name + '/',shell=True)
    subprocess.run('mv *h5 ' + icesat2_dir+city_name + '/',shell=True)
    subprocess.run('rm response-header.txt',shell=True)
    subprocess.run('unzip \'' + icesat2_dir+city_name + '/*zip\' -d ' + icesat2_dir+city_name + '/',shell=True)
    subprocess.run('mv ' + icesat2_dir+city_name + '/*/processed*.h5 ' + icesat2_dir+city_name+ '/',shell=True)
    [os.rmdir(os.path.join(icesat2_dir,city_name,sub_dir)) for sub_dir in os.listdir(os.path.join(icesat2_dir,city_name)) if os.path.isdir(os.path.join(icesat2_dir,city_name,sub_dir)) and len(os.listdir(os.path.join(icesat2_dir,city_name,sub_dir)))==0]
    subprocess.run('rm ' + icesat2_dir + city_name + '/*.json',shell=True)
    subprocess.run('rm ' + icesat2_dir+city_name + '/*zip',shell=True)
    subprocess.run('find ' + icesat2_dir+city_name + '/*h5 -printf "%f\\'+'n" > ' + icesat2_dir+city_name + '/icesat2_list.txt',shell=True)
    return None

def analyze_icesat2(icesat2_dir,df_city,shp_data):
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
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    h = np.empty([0,1],dtype=float)
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
            tmp_lon_high_conf = tmp_lon[tmp_high_conf]
            tmp_lat_high_conf = tmp_lat[tmp_high_conf]
            tmp_h_high_conf = tmp_h[tmp_high_conf]
            tmp_delta_time_total_high_conf = tmp_delta_time_total[tmp_high_conf]
            lon = np.append(lon,tmp_lon)
            lat = np.append(lat,tmp_lat)
            h = np.append(h,tmp_h)
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

def landmask_icesat2(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag=1):
    #Given lon/lat of points, and lon/lat of coast (or any other boundary),
    #finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    #with polygons separated by NaNs
    print('Running landmask...')
    t_start = datetime.datetime.now()
    c_float_p = c.POINTER(c.c_float)
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    subprocess.run('cc -fPIC -shared -o ' + landmask_so_file + ' ' + landmask_c_file,shell=True)
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    arrx = (c.c_float * len(lon_coast))(*lon_coast)
    arry = (c.c_float * len(lat_coast))(*lat_coast)
    arrx_input = (c.c_float * len(lon))(*lon)
    arry_input = (c.c_float * len(lat))(*lat)
    landmask = np.zeros(len(lon),dtype=c.c_int)
    landmask_lib.pnpoly(c.c_int(len(lon_coast)),c.c_int(len(lon)),arrx,arry,arrx_input,arry_input,c.c_void_p(landmask.ctypes.data))
    landmask = landmask == inside_flag #just to be consistent and return Boolean array
    t_end = datetime.datetime.now()
    print('Landmask done.')
    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    dt_hour, dt_min = divmod(dt_min,60)
    print('It took:')
    print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
    return landmask

def SRTM_filter_icesat2(lon,lat,h,icesat2_file,icesat2_dir,df_city,user,pw,SRTM_threshold,EGM96_path):
    #Given lon/lat/h, downloads SRTM 1x1 deg tiles, merges tiles together and changes from referencing EGM96 to WGS84 ellipsoid.
    #Then, samples the lon/lat points of ICESat-2 at the full SRTM mosaic
    #If the difference between the ICESat-2 height and SRTM height is larger than the given threshold, discard that ICESat-2 point
    #This is primarily a good way to get rid of large outliers, e.g. due to clouds
    city_name = df_city.city
    lon_min_SRTM = np.min(lon)
    lon_max_SRTM = np.max(lon)
    lat_min_SRTM = np.min(lat)
    lat_max_SRTM = np.max(lat)
    SRTM_list = []
    lon_range = range(int(np.floor(lon_min_SRTM)),int(np.floor(lon_max_SRTM))+1)
    lat_range = range(int(np.floor(lat_min_SRTM)),int(np.floor(lat_max_SRTM))+1)
    for i in range(len(lon_range)):
        for j in range(len(lat_range)):
            if lon_range[i] >= 0:
                lonLetter = 'E'
            else:
                lonLetter = 'W'
            if lat_range[j] >= 0:
                latLetter = 'N'
            else:
                latLetter = 'S'
            lonCode = f"{int(np.abs(np.floor(lon_range[i]))):03d}"
            latCode = f"{int(np.abs(np.floor(lat_range[j]))):02d}"
            SRTM_id = latLetter + latCode + lonLetter + lonCode
            SRTM_list.append(SRTM_id)
    merge_command = 'gdal_merge.py -q -o ' + icesat2_dir + city_name + '/' + city_name + '_SRTM.tif '
    print('Downloading SRTM...')
    for i in range(len(SRTM_list)):
        DL_command = 'wget --user=' + user + ' --password=' + pw + ' https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/' + SRTM_list[i] + '.SRTMGL1.hgt.zip --no-check-certificate --quiet'
        subprocess.run(DL_command,shell=True)
        exists = os.path.isfile(SRTM_list[i] + '.SRTMGL1.hgt.zip')
        if exists:
            mv_command = 'mv ' + SRTM_list[i] + '.SRTMGL1.hgt.zip ' + icesat2_dir + city_name + '/'
            unzip_command = 'unzip -qq ' + icesat2_dir + city_name + '/' + SRTM_list[i] + '.SRTMGL1.hgt.zip -d ' + icesat2_dir + city_name + '/'
            delete_command = 'rm ' + icesat2_dir + city_name + '/' + SRTM_list[i] + '.SRTMGL1.hgt.zip'
            subprocess.run(mv_command,shell=True)
            subprocess.run(unzip_command,shell=True)
            subprocess.run(delete_command,shell=True)
            merge_command = merge_command + icesat2_dir + city_name + '/' + SRTM_list[i] + '.hgt '
    print('Downloaded SRTM.')
    print('Merging SRTM...')
    subprocess.run(merge_command,shell=True)
    print('Merged SRTM.')
    src = gdal.Open(EGM96_path, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    match_filename = icesat2_dir + city_name + '/' + city_name + '_SRTM.tif'
    match_ds = gdal.Open(match_filename,gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    dst_filename = icesat2_dir + city_name + '/EGM96_' + city_name + '.tif'
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)
    del dst
    SRTM_wgs84_file = icesat2_dir + city_name + '/' + city_name + '_SRTM_WGS84.tif'
    subprocess.run('gdal_calc.py -A ' + dst_filename + ' -B ' + match_filename + ' --outfile ' + SRTM_wgs84_file + ' --calc=A+B --quiet',shell=True)
    subprocess.run('rm ' + match_filename,shell=True)
    subprocess.run('rm ' + dst_filename,shell=True)
    for jj in range(len(SRTM_list)):
        subprocess.run('rm ' + icesat2_dir + city_name + '/' + SRTM_list[jj] + '.hgt',shell=True)
    SRTM_sampled_file = icesat2_dir + city_name + '/' + city_name + '_sampled_SRTM.txt'
    print('Sampling SRTM...')
    subprocess.run('cut -d\',\' -f1-2 ' + icesat2_file + ' | sed \'s/,/ /g\' | gdallocationinfo -wgs84 -valonly ' + SRTM_wgs84_file + ' > ' + SRTM_sampled_file,shell=True)
    print('Sampled SRTM')
    df_SRTM = pd.read_csv(SRTM_sampled_file,header=None,names=['h_SRTM'],dtype={'h_SRTM':'float'})
    h_sampled_SRTM = np.asarray(df_SRTM.h_SRTM)
    SRTM_cond = np.abs(h - h_sampled_SRTM) < SRTM_threshold
    return SRTM_cond

def main():
    SRTM_toggle = True
    landmask_toggle = True
    timestamp_toggle = True
    on_off_str = ('off','on')

    print('Current settings:')
    print(f'SRTM filtering : {on_off_str[SRTM_toggle]}')
    print(f'Landmask       : {on_off_str[landmask_toggle]}')
    print(f'Timestamps     : {on_off_str[timestamp_toggle]}')

    input_file = '/home/eheijkoop/INPUTS/GCP_Input.txt' #Input file with location name,lon_min,lon_max,lat_min,lat_max (1 header line)
    osm_shp_path = '/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/Coast/land-polygons-complete-4326/land_polygons.shp' #OpenStreetMap land polygons, available at https://osmdata.openstreetmap.de/data/land-polygons.html (use WGS84, not split)
    icesat2_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/ICESat-2/' #output directory, which will be populated by subdirectories named after your input
    error_log_file = icesat2_dir + 'ICESat2_Log_File.txt'
    landmask_c_file = '/home/eheijkoop/Scripts/C_Code/pnpoly_function.c' #file with C function pnpoly, "point in polygon", to perform landmask
    landmask_inside_flag = 1 #flag to find points inside (1) or outside (0) polygon

    user = 'EHeijkoop' #Your NASA EarthData username
    token = get_token(user) #Create NSIDC token to download ICESat-2
    if SRTM_toggle:
        pw = getpass.getpass() #Your NASA EarthData password
        SRTM_threshold = 10 #set your SRTM threshold here
        EGM96_path = '/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/EGM96_180.tif' #supplied on github
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
        lon_high_conf,lat_high_conf,h_high_conf,delta_time_total_high_conf = analyze_icesat2(icesat2_dir,df_extents.iloc[i],shp_data)
        if len(lon_high_conf) == 0:
            continue
        utc_time_high_conf = gps2utc(delta_time_total_high_conf)
        if landmask_toggle == True:
            landmask = landmask_icesat2(lon_high_conf,lat_high_conf,lon_coast,lat_coast,landmask_c_file,landmask_inside_flag)
            lon_high_conf = lon_high_conf[landmask]
            lat_high_conf = lat_high_conf[landmask]
            h_high_conf = h_high_conf[landmask]
            delta_time_total_high_conf = delta_time_total_high_conf[landmask]
            utc_time_high_conf = gps2utc(delta_time_total_high_conf)
            icesat2_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf_masked.txt'
        else:
            icesat2_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf.txt'
        
        if timestamp_toggle:
            np.savetxt(icesat2_file,np.c_[lon_high_conf,lat_high_conf,h_high_conf],fmt='%10.5f,%10.5f,%10.5f',delimiter=',')
        else:
            np.savetxt(icesat2_file,np.c_[lon_high_conf,lat_high_conf,h_high_conf,utc_time_high_conf.astype(object)],fmt='%10.5f,%10.5f,%10.5f,%s',delimiter=',')
        
        if SRTM_toggle:
            SRTM_cond = SRTM_filter_icesat2(lon_high_conf,lat_high_conf,h_high_conf,icesat2_file,icesat2_dir,df_extents.iloc[i],user,pw,SRTM_threshold,EGM96_path)
            lon_high_conf_SRTM = lon_high_conf[SRTM_cond]
            lat_high_conf_SRTM = lat_high_conf[SRTM_cond]
            h_high_conf_SRTM = h_high_conf[SRTM_cond]
            delta_time_total_high_conf_SRTM = delta_time_total_high_conf[SRTM_cond]
            utc_time_high_conf_SRTM = gps2utc(delta_time_total_high_conf_SRTM)
            if landmask_toggle:
                icesat2_srtm_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf_masked_SRTM_filtered_threshold_' + str(SRTM_threshold) + '_m.txt'
            else:
                icesat2_srtm_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf_SRTM_filtered_threshold_' + str(SRTM_threshold) + '_m.txt'
            if timestamp_toggle:
                np.savetxt(icesat2_srtm_file,np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM,utc_time_high_conf_SRTM.astype(object)],fmt='%10.5f,%10.5f,%10.5f,%s',delimiter=',')
            else:
                np.savetxt(icesat2_srtm_file,np.c_[lon_high_conf_SRTM,lat_high_conf_SRTM,h_high_conf_SRTM],fmt='%10.5f,%10.5f,%10.5f',delimiter=',')
        print('Done with '+city_name+' at '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        print(' ')


if __name__ == '__main__':
    main()
