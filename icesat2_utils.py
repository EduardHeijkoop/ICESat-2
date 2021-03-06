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


def get_lonlat_shp(shp_path):
    '''
    Given a shapefile (.shp), returns longitude and latitude arrays
    of all individual polygons, separated by NaNs
    Polygons within polygons will be included here
    '''
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

def great_circle_distance(lon1,lat1,lon2,lat2,R=6378137.0):
    lon1 = deg2rad(lon1)
    lat1 = deg2rad(lat1)
    lon2 = deg2rad(lon2)
    lat2 = deg2rad(lat2)
    DL = np.abs(lon2 - lon1)
    DP = np.abs(lat2 - lat1)
    dsigma = 2*np.arcsin( np.sqrt( np.sin(0.5*DP)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*DL)**2))
    distance = R*dsigma
    return distance

def deg2rad(deg):
    rad = deg*np.math.pi/180
    return rad

def gps2utc(gps_time):
    '''
    Converts GPS time that ICESat-2 references to UTC
    '''
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

def get_token(user):
    '''
    Given username and password, get NSIDC token to download ICESat-2
    Checks age of token file as it expires after 30 days
    If token not present, triggers next if statement immediately to generate new token
    '''
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
    '''
    Cleans up a number of files that linger after download
    '''
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

def move_icesat2(icesat2_dir,df_city):
    '''
    move .zip files that are downloaded by the download_icesat2 function to the correct directory
    unzip them there and put all .h5 files in that directory
    remove empty subdirectories that linger after unzipping
    remove zip files
    create icesat2_list.txt file with list of .h5 files 
    '''
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


def get_osm_extents(df_city,osm_shp_path,icesat2_dir):
    '''
    Given lon/lat extents in a Pandas DataFrame (df_city),
    subsets OpenStreetMap land polygons to those extents as a new shapefile
    and returns lon/lat of that new shapefile 
    '''
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
    '''
    Validates date input
    '''
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

def cat_str_API(beam):
    '''
    Strings together hdf5 path for:
        photon h/lon/lat/signal_conf
        reference photon lon/lat/sigma_h/sigma_lon/sigma_lat
        ocean tide/DAC
        (geophysical corrections') delta_time
    For all 3 strong beams
    '''
    beam_command = '/gt'+beam+'/heights/h_ph,/gt'+beam+'/heights/lon_ph,/gt'+beam+'/heights/lat_ph,/gt'+beam+'/heights/delta_time,/gt'+beam+'/heights/signal_conf_ph,' \
        '/gt'+beam+'/geolocation/reference_photon_lon,/gt'+beam+'/geolocation/reference_photon_lat,/gt'+beam+'/geolocation/ph_index_beg,/gt'+beam+'/geolocation/segment_ph_cnt,/gt'+beam+'/geolocation/reference_photon_index,' \
                '/gt'+beam+'/geolocation/sigma_h,/gt'+beam+'/geolocation/sigma_lon,/gt'+beam+'/geolocation/sigma_lat,' \
        '/gt'+beam+'/geophys_corr/delta_time,/gt'+beam+'/geophys_corr/tide_ocean,/gt'+beam+'/geophys_corr/dac,/gt'+beam+'/geophys_corr/tide_equilibrium,'
    return beam_command


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

def landmask_icesat2(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag):
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
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

def DTU21_filter_icesat2(h_unfiltered,icesat2_file,icesat2_dir,df_city,DTU21_threshold,DTU21_path):
    #Given an ICESat-2 file, samples the DTU21 Mean Sea Surface file at the lon/lat points of ICESat-2
    #If the difference between the ICESat-2 height and DTU21 height is larger than the given threshold, discard that ICESat-2 point
    #This is primarily a good way to get rid of large outliers, e.g. due to clouds
    city_name = df_city.city
    DTU21_sampled_file = icesat2_dir + city_name + '/' + city_name + '_sampled_DTU21.txt'
    print('Sampling DTU21...')
    subprocess.run('cut -d\',\' -f1-2 ' + icesat2_file + ' | sed \'s/,/ /g\' | gdallocationinfo -wgs84 -valonly ' + DTU21_path + ' > ' + DTU21_sampled_file,shell=True)
    print('Sampled DTU21')
    df_DTU21 = pd.read_csv(DTU21_sampled_file,header=None,names=['h_DTU21'],dtype={'h_DTU21':'float'})
    h_sampled_DTU21 = np.asarray(df_DTU21.h_DTU21)
    DTU21_cond = np.abs(h_unfiltered - h_sampled_DTU21) < DTU21_threshold
    subprocess.run('rm ' + DTU21_sampled_file,shell=True)
    return DTU21_cond


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
    merge_command = f'gdal_merge.py -q -o {icesat2_dir}{city_name}/{city_name}_SRTM.tif '
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
    match_ds = gdal.Open(match_filename,gdalconst.GA_Update)
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
    SRTM_sampled_file = f'{icesat2_dir}{city_name}/{city_name}_sampled_SRTM.txt'
    print('Sampling SRTM...')
    subprocess.run(f'cut -d\',\' -f1-2 {icesat2_file} | sed \'s/,/ /g\' | gdallocationinfo -wgs84 -valonly {SRTM_wgs84_file} > {SRTM_sampled_file}',shell=True)
    print('Sampled SRTM')
    df_SRTM = pd.read_csv(SRTM_sampled_file,header=None,names=['h_SRTM'],dtype={'h_SRTM':'float'})
    h_sampled_SRTM = np.asarray(df_SRTM.h_SRTM)
    SRTM_cond = np.abs(h - h_sampled_SRTM) < SRTM_threshold
    subprocess.run(f'rm {SRTM_sampled_file}',shell=True)
    subprocess.run(f'rm {SRTM_wgs84_file}',shell=True)
    return SRTM_cond
