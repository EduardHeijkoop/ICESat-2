import os
import glob
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
import itertools
import multiprocessing
import json
import requests
import time
import sys
import itertools

def get_lonlat_shp(shp_path):
    '''
    Given a shapefile (.shp), returns longitude and latitude arrays
    of all individual polygons, separated by NaNs
    Polygons within polygons will be included here
    Deprecated as of shapely version 2.0
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

def get_lonlat_geometry(geom):
    '''
    Returns lon/lat of all exteriors and interiors of a Shapely geomery:
        -Polygon
        -MultiPolygon
        -GeometryCollection
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    if geom.geom_type == 'Polygon':
        lon_geom,lat_geom = get_lonlat_polygon(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'GeometryCollection':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    return lon,lat

def get_lonlat_polygon(polygon):
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    exterior_xy = np.asarray(polygon.exterior.xy)
    lon = np.append(lon,exterior_xy[0,:])
    lon = np.append(lon,np.nan)
    lat = np.append(lat,exterior_xy[1,:])
    lat = np.append(lat,np.nan)
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords.xy)
        lon = np.append(lon,interior_xy[0,:])
        lon = np.append(lon,np.nan)
        lat = np.append(lat,interior_xy[1,:])
        lat = np.append(lat,np.nan)
    return lon,lat

def get_lonlat_gdf(gdf):
    '''
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame.
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

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
    gps_time = np.atleast_1d(gps_time)
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

def utc2gps(utc_time_str):
    '''
    Converts UTC time to GPS time that ICESat-2 references
    '''
    utc_time_str = np.atleast_1d(utc_time_str)
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    utc_time = np.asarray([datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f') for x in utc_time_str])
    gps_time_datetime = utc_time - leap_seconds*datetime.timedelta(seconds=1)
    gps_time = np.asarray([(x-t0).total_seconds() for x in gps_time_datetime])
    return gps_time

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

# def user_pw_to_base64(user,pw):
#     '''
#     Converts username and password to base64 encoding
#     '''
#     user_pw = user+':'+pw
#     user_pw_bytes = user_pw.encode('ascii')
#     user_pw_base64 = base64.b64encode(user_pw_bytes)
#     user_pw_base64 = user_pw_base64.decode('ascii')
#     return user_pw_base64

# def get_token(user):
#     '''
#     Given NASA EarthData username, download NASA URS token.
#     Token expires after 60 days
#     Checks if there are any tokens available and if so, if they have expired.
#     If they are expired, deletes them, and generates a new token.
#     '''
#     today_datetime = datetime.datetime.now()
#     if os.path.isfile('Token.json'):
#         token_data = json.loads(open('Token.json').read())
#         expiration_date = datetime.datetime.strptime(token_data['expiration_date'],'%m/%d/%Y')



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
    city_dir = f'{icesat2_dir}{city_name}/'
    subprocess.run(f'mv *zip {city_dir}',shell=True)
    zip_list = sorted(glob.glob(f'{city_dir}*.zip'))
    for zip_file in zip_list:
        subprocess.run(f'unzip -q \'{zip_file}\' -d {city_dir}',shell=True)
        subprocess.run(f'mv {city_dir}*/processed_ATL03*h5 {city_dir}',shell=True)
        [os.rmdir(os.path.join(icesat2_dir,city_name,sub_dir)) for sub_dir in os.listdir(os.path.join(icesat2_dir,city_name)) if os.path.isdir(os.path.join(icesat2_dir,city_name,sub_dir)) and len(os.listdir(os.path.join(icesat2_dir,city_name,sub_dir)))==0]
        subprocess.run(f'rm {zip_file}',shell=True)
        if os.path.isfile(f'{city_dir}README'):
            subprocess.run(f'rm {city_dir}README',shell=True)
        json_list = sorted(glob.glob(f'{city_dir}request*.json'))
        if len(json_list) > 0:
            subprocess.run(f'rm {city_dir}request*.json',shell=True)
    subprocess.run(f'find {city_dir}*h5 -printf "%f\\n" > {city_dir}icesat2_list.txt',shell=True)
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
    lon_coast,lat_coast = get_lonlat_gdf(shp_data)
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
        photon h/lon/lat/signal_conf/quality
        reference photon lon/lat/sigma_h/sigma_lon/sigma_lat
        ocean tide/DAC
        (geophysical corrections') delta_time
    For all 3 strong beams
    '''
    beam_command = f'/{beam}/heights/h_ph,/{beam}/heights/lon_ph,/{beam}/heights/lat_ph,/{beam}/heights/delta_time,/{beam}/heights/signal_conf_ph,/{beam}/heights/quality_ph,/{beam}/heights/weight_ph,' \
        f'/{beam}/geolocation/reference_photon_lon,/{beam}/geolocation/reference_photon_lat,/{beam}/geolocation/ph_index_beg,/{beam}/geolocation/segment_ph_cnt,/{beam}/geolocation/reference_photon_index,' \
        f'/{beam}/geolocation/sigma_h,/{beam}/geolocation/sigma_lon,/{beam}/geolocation/sigma_lat,/{beam}/geolocation/podppd_flag,' \
        f'/{beam}/geophys_corr/delta_time,/{beam}/geophys_corr/tide_ocean,/{beam}/geophys_corr/dac,/{beam}/geophys_corr/tide_equilibrium,'
    return beam_command


def download_icesat2(user,pw,df_city,version):
    version_str = f'{version:03d}'
    short_name = 'ATL03'
    dl_path = os.getcwd()
    symbol_list = ['|','/','-','\\','|','/','-','\\']
    cmr_params = {'short_name':short_name}
    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    cmr_response = requests.get(cmr_collections_url, params=cmr_params)
    cmr_results = json.loads(cmr_response.content)
    available_versions = [entr['version_id'] for entr in cmr_results['feed']['entry']]
    if not version_str in available_versions:
        print(f'Version {version_str} not available. Available versions are: {available_versions}')
        return None
    
    city_name = df_city.city
    t_start = df_city.t_start
    t_end = df_city.t_end
    t_start_valid = validate_date(t_start)
    t_end_valid = validate_date(t_end)
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    #Spatial bounding box:
    bbox = f'{df_city.lon_min},{df_city.lat_min},{df_city.lon_max},{df_city.lat_max}'
    bounding_box = bbox
    #Temporal bounds:
    if t_start_valid == True:
        t_start = datetime.datetime.strptime(t_start,'%Y-%m-%d')
        t_start_year = t_start.year
        t_start = t_start.strftime('%Y-%m-%d')
    else:
        t_start = '2018-10-01'
        t_start_year = 2018
    if t_end_valid == True:
        t_end = datetime.datetime.strptime(t_end,'%Y-%m-%d')
        t_end_year = t_end.year
        t_end = t_end.strftime('%Y-%m-%d')
    else:
        t_end = datetime.datetime.now().strftime('%Y-%m-%d')
        t_end_year = datetime.datetime.now().year
    for year in np.arange(t_start_year,t_end_year+1):
        t_start_command = f'{year}-01-01'
        t_end_command = f'{year}-12-31'
        if year == t_start_year:
            t_start_command = t_start
        if year == t_end_year:
            t_end_command = t_end
        time_command = f'{t_start_command}T00:00:00Z,{t_end_command}T23:59:59Z'
        # if np.logical_and(t_start_valid==False,t_end_valid==False):
        #     time_command = ''
        # else:
        #     time_command = f'{t_start}T00:00:00Z,{t_end}T23:59:59Z'
        #Coverage in terms of variables you want to download.
        coverage_command = ''
        beam_list = ['gt1l','gt1r','gt2l','gt2r','gt3l','gt3r']
        for beam in beam_list:
            coverage_command = coverage_command + cat_str_API(beam)
        coverage_command = coverage_command + '/orbit_info/sc_orient,/ancillary_data/atlas_sdp_gps_epoch,/ancillary_data/data_start_utc,/ancillary_data/data_end_utc'

        search_params = {
            'short_name': short_name,
            'version': version_str,
            'temporal': time_command,
            'page_size': 100,
            'page_num': 1,
            'bounding_box': bbox
        }
        granules = []
        headers={'Accept': 'application/json'}
        while True:
            response = requests.get(granule_search_url, params=search_params, headers=headers)
            results = json.loads(response.content)
            if len(results['feed']['entry']) == 0:
                break
            granules.extend(results['feed']['entry'])
            search_params['page_num'] += 1
        granules = [granule for granule in granules if f'SC:ATL03.{version_str}' in granule['title']]
        print(f'There are {len(granules)} granules of {short_name} over {city_name} for the year {year}.')

        # granule_sizes = [float(granule['granule_size']) for granule in granules]
        # dl_size = np.sum(granule_sizes)
        # file_size_ext = 'MB'
        # if dl_size > 1024:
        #     dl_size = dl_size / 1024
        #     file_size_ext = 'GB'
        # print(f'Total download size before spatial subsetting: {dl_size:.1f} {file_size_ext}')

        capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{version_str}.xml'
        capability_session = requests.session()
        capability_s = capability_session.get(capability_url)
        capability_response = capability_session.get(capability_s.url,auth=(user,pw))
        capability_root = ET.fromstring(capability_response.content)
        subagent = [subset_agent.attrib for subset_agent in capability_root.iter('SubsetAgent')]
        if len(subagent) == 0:
            print('No subset agent found.')
            return None
        spatial_subsetting_flag = bool(subagent[0]['spatialSubsetting'])
        if spatial_subsetting_flag == False:
            print('Spatial subsetting not available.')
            return None
        N_sync = int(subagent[0]['maxGransSyncRequest'])
        N_async = int(subagent[0]['maxGransAsyncRequest'])

        if len(granules) > N_sync:
            request_mode = 'async'
            page_size = 100 #for some reason higher values will not download all data (more recent files will be excluded)
            sync_async_code = 'async'
            print('Going for asynchronous request.')
        else:
            request_mode = 'stream'
            page_size = N_sync
            sync_async_code = 'sync'
            print('Going for synchronous request.')
        page_num = int(np.ceil(len(granules)/page_size))

        #empty parameters will be deleted, but can be added at user's discretion
        param_dict = {'short_name': short_name, 
                    'version': version_str, 
                    'temporal': time_command, 
                    'time': time_command.replace('Z',''), 
                    'bounding_box': bounding_box, 
                    'bbox': bbox, 
                    'format': '', 
                    'projection': '', 
                    'projection_parameters': '',
                    'Coverage': coverage_command, 
                    'page_size': page_size, 
                    'request_mode': request_mode, 
                    'agent': '', 
                    'email': '', }
        param_dict = {k: v for k, v in param_dict.items() if v != ''}
        param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
        param_string = param_string.replace("'","")

        endpoint_list = [] 
        for i in range(page_num):
            page_val = i + 1
            API_request = f'{base_url}?{param_string}&page_num={page_val}'
            endpoint_list.append(API_request)

        print(f'Downloading for {city_name}...')
        if request_mode=='async':
            for i in range(page_num):
                page_val = i + 1
                param_dict['page_num'] = page_val
                request = capability_session.get(base_url, params=param_dict)
                request.raise_for_status()
                esir_root = ET.fromstring(request.content)
                orderlist = []   
                for order in esir_root.findall("./order/"):
                    orderlist.append(order.text)
                orderID = orderlist[0]
                statusURL = base_url + '/' + orderID
                request_response = capability_session.get(statusURL)    
                request_response.raise_for_status()
                request_root = ET.fromstring(request_response.content)
                statuslist = []
                for status in request_root.findall("./requestStatus/"):
                    statuslist.append(status.text)
                status = statuslist[0]
                print(f'Processing {page_val}/{page_num} at NSIDC...')
                symbol_count = -1
                while status == 'pending' or status == 'processing': 
                    symbol_count += 1
                    idx = np.mod(symbol_count,len(symbol_list))
                    sym = symbol_list[idx]
                    sys.stdout.write('\r')
                    sys.stdout.write(sym)
                    sys.stdout.flush()
                    time.sleep(10)
                    loop_response = capability_session.get(statusURL)
                    loop_response.raise_for_status()
                    loop_root = ET.fromstring(loop_response.content)
                    statuslist = []
                    for status in loop_root.findall("./requestStatus/"):
                        statuslist.append(status.text)
                    status = statuslist[0]
                    if status == 'pending' or status == 'processing':
                        continue
                if status == 'complete_with_errors' or status == 'failed':
                    messagelist = []
                    print('Error messages:')
                    for message in loop_root.findall("./processInfo/"):
                        print(message.text)
                        messagelist.append(message.text)
                print('\n')
            # Download zipped order if status is complete or complete_with_errors
                if status == 'complete' or status == 'complete_with_errors':
                    downloadURL = f'https://n5eil02u.ecs.nsidc.org/esir/{orderID}.zip'
                    print(f'Downloading file {page_val}/{page_num}...')
                    zip_response = capability_session.get(downloadURL)
                    zip_response.raise_for_status()
                    fz = open(f'{dl_path}/{city_name}_{year}_{page_val}.zip', 'wb')
                    fz.write(zip_response.content)
                    fz.close()
                else:
                    print('Request failed.')
        else:
            for i in range(page_num):
                page_val = i + 1
                print(f'Downloading file {page_val}/{page_num}...')
                param_dict['page_num'] = page_val
                request = capability_session.get(base_url, params=param_dict)
                request.raise_for_status()
                fz = open(f'{dl_path}/{city_name}_{year}_{page_val}.zip', 'wb')
                fz.write(request.content)
                fz.close()
        print('Download complete.')

    return sync_async_code

# def download_icesat2(df_city,token,error_log_file,version=5):
#     #Given lon/lat extents in a Pandas DataFrame (df_city),
#     #downloads ICESat-2 ATL03 geolocated photons
#     city_name = df_city.city
#     t_start = df_city.t_start
#     t_end = df_city.t_end
#     t_start_valid = validate_date(t_start)
#     t_end_valid = validate_date(t_end)
#     lon_min_str = str(df_city.lon_min)
#     lon_max_str = str(df_city.lon_max)
#     lat_min_str = str(df_city.lat_min)
#     lat_max_str = str(df_city.lat_max)
#     token_command = 'token='+token
#     site_command = 'https://n5eil02u.ecs.nsidc.org/egi/request?'
#     email_command = 'email=false'
#     short_name = 'ATL03'
#     coverage_command = 'coverage='
#     beam_list = ['1l','1r','2l','2r','3l','3r']
#     for beam in beam_list:
#         coverage_command = coverage_command + cat_str_API(beam)
#     coverage_command = coverage_command + '/orbit_info/sc_orient,/ancillary_data/atlas_sdp_gps_epoch,/ancillary_data/data_start_utc,/ancillary_data/data_end_utc'
#     short_name_command = f'short_name={short_name}&version={version:03d}'
#     if t_start_valid == True:
#         t_start = datetime.datetime.strptime(t_start,'%Y-%m-%d').strftime('%Y-%m-%d')
#     else:
#         t_start = '2018-10-01'

#     if t_end_valid == True:
#         t_end = datetime.datetime.strptime(t_end,'%Y-%m-%d').strftime('%Y-%m-%d')
#     else:
#         t_end = datetime.datetime.now().strftime('%Y-%m-%d')

#     if np.logical_and(t_start_valid==False,t_end_valid==False):
#         time_command = ''
#     else:
#         time_command = 'time='+t_start+'T00:00:00,'+t_end+'T23:59:59&'
#     bounding_box_command = 'bounding_box='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
#     bbox_command = 'bbox='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
#     shape_command = bounding_box_command + '&' + bbox_command + '&'
#     page_number = 1
#     page_condition = True
#     while page_condition:
#         page_command = 'page_num='+str(page_number)
#         full_command = 'curl -O -J -k --dump-header response-header.txt \"' + \
#             site_command + '&' + short_name_command + '&' + token_command + '&' + \
#             email_command + '&' + shape_command + time_command + \
#             coverage_command + '&' + page_command + '\"'
#         subprocess.run(full_command,shell=True)
#         with open('response-header.txt','r') as f2:
#             response_line = f2.readline().replace('\n','')
#         if response_line[9:12] == '200':
#             page_number = page_number + 1
#         elif response_line[9:12] == '204':
#             print('End of download.')
#             page_condition = False
#         elif response_line[9:12] == '501':
#             page_condition = False
#         else:
#             print('Something bad happened.')
#             print('Exiting...')
#             page_condition = False
#     if page_number == 1:
#         print('Nothing was downloaded.')
#         print('Check extents - possibly no coverage!')
#         now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(error_log_file,'a') as text_file:
#             text_file.write(now_str + ': ' + city_name + ' - No data download.\n')
#             print('No data downloaded!')
#         return 0
#     else:
#         return None

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

def parallel_landmask(lon_pts,lat_pts,lon_boundary,lat_boundary,landmask_c_file,inside_flag,N_cpus=1):
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    if not os.path.exists(landmask_so_file):
        subprocess.run('cc -fPIC -shared -o ' + landmask_so_file + ' ' + landmask_c_file,shell=True)
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    lon_split = np.array_split(lon_pts,N_cpus)
    lat_split = np.array_split(lat_pts,N_cpus)
    ir = itertools.repeat
    p = multiprocessing.Pool(N_cpus)
    landmask = p.starmap(parallel_pnpoly,zip(lon_split,lat_split,ir(lon_boundary),ir(lat_boundary),ir(landmask_so_file)))
    p.close()
    landmask = np.concatenate(landmask)
    landmask = landmask == inside_flag
    return landmask

def parallel_pnpoly(lon_pts,lat_pts,lon_boundary,lat_boundary,landmask_so_file):
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    arrx = (c.c_float * len(lon_boundary))(*lon_boundary)
    arry = (c.c_float * len(lat_boundary))(*lat_boundary)
    arrx_input = (c.c_float * len(lon_pts))(*lon_pts)
    arry_input = (c.c_float * len(lat_pts))(*lat_pts)
    landmask = np.zeros(len(lon_pts),dtype=c.c_int)
    landmask_lib.pnpoly(c.c_int(len(lon_boundary)),c.c_int(len(lon_pts)),arrx,arry,arrx_input,arry_input,c.c_void_p(landmask.ctypes.data))
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

def delta_time_to_orientation(delta_time):
    '''
    Converts delta time array to array of orientations, i.e. forward or backwards
    Will update as yaw maneuvers happen
    '''
    delta_time = np.atleast_1d(delta_time)
    yaw_maneuvers = [datetime.datetime(2018,9,15,13,2,0),
                    datetime.datetime(2018,12,28,18,53,8),
                    datetime.datetime(2019,9,7,1,4,6),
                    datetime.datetime(2020,5,14,1,49,3),
                    datetime.datetime(2021,1,15,15,17,1),
                    datetime.datetime(2021,10,2,2,20,1),
                    datetime.datetime(2022,6,9,1,31,19),
                    datetime.datetime(2023,2,9,16,59,14),
                    datetime.datetime(2023,10,27,13,26,50)]
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18
    yaw_maneuvers_gps = np.asarray(yaw_maneuvers) - leap_seconds*datetime.timedelta(seconds=1)
    delta_time_yaw_maneuvers = np.asarray([(x-t0).total_seconds() for x in yaw_maneuvers_gps])
    #check that all delta times are in between the same yaw maneuvers
    if np.sum(delta_time[0] > delta_time_yaw_maneuvers) == np.sum(delta_time[-1] > delta_time_yaw_maneuvers):
        sc_orient = np.mod(np.sum(delta_time[0] > delta_time_yaw_maneuvers),2) * np.ones(len(delta_time),dtype=int)
    else:
        #very unlikely, but if delta time crosses over yaw maneuver, then we need to find where and assign the right values
        sc_orient = np.mod(np.asarray([np.sum(dt > delta_time_yaw_maneuvers) for dt in delta_time]),2)
    return sc_orient

def beam_orientation_to_strength(beam,orientation):
    '''
    Based on beam (e.g. gt2r) and orientation (0 or 1), returns the strength of the beam (strong or weak)
    '''
    beam = np.atleast_1d(beam)
    orientation = np.atleast_1d(orientation)
    beam_options = ['r','l']
    orientation_options = [0,1]
    if len(beam) != len(orientation):
        print('Beam and orientation arrays must be the same length.')
        return None
    beam_side = np.asarray([b[3] for b in beam])
    strength = np.zeros(len(beam),dtype='<U1')
    idx_strong = np.logical_or(np.logical_and(orientation==0,beam_side=='l'),np.logical_and(orientation==1,beam_side=='r'))
    idx_weak = np.logical_or(np.logical_and(orientation==0,beam_side=='r'),np.logical_and(orientation==1,beam_side=='l'))
    strength[idx_strong] = 's'
    strength[idx_weak] = 'w'
    return strength