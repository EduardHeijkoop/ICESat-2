import ee
import numpy as np
import geopandas as gpd
import pandas as pd
import os,sys
import subprocess
import datetime
import shapely
import configparser
import argparse
import warnings
import time
import ctypes as c
import glob
import io
import multiprocessing
import itertools

import google.auth.transport.requests #Request
import google.oauth2.credentials #Credentials
import google_auth_oauthlib.flow #InstalledAppFlow
import googleapiclient.discovery #build
import googleapiclient.errors #HttpError
import googleapiclient.http #MediaIoBaseDownload

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

def landmask_csv(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag):
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
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
    return landmask

def get_NDVI(s2_image):
    ndvi = s2_image.normalizedDifference(['B8','B4']).rename('NDVI')
    return ndvi

def get_NDWI(s2_image):
    ndwi = s2_image.normalizedDifference(['B3','B8']).rename('NDWI')
    return ndwi

def getANDWI(s2_image):
    red = s2_image.select('B4')
    green = s2_image.select('B3')
    blue = s2_image.select('B2')
    nir = s2_image.select('B8')
    swir1 = s2_image.select('B11')
    swir2 = s2_image.select('B12')
    andwi = (red.add(green).add(blue).subtract(nir).subtract(swir1).subtract(swir2)).divide(red.add(green).add(blue).add(nir).add(swir1).add(swir2)).rename('ANDWI')
    return andwi

def get_NDSI(s2_image):
    ndsi = s2_image.normalizedDifference(['B11','B3']).rename('NDSI')
    return ndsi

def get_Optical(s2_image):
    optical = s2_image.select('B2','B3','B4').rename('Optical')
    return optical

def get_FalseColor(s2_image):
    false_color = s2_image.select('B3','B4','B8').rename('False_Color')
    return false_color

def get_NDVI_threshold(s2_image,NDVI_THRESHOLD):
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD)

def get_NDWI_threshold(s2_image,NDWI_THRESHOLD):
    return s2_image.normalizedDifference(['B3','B8']).gt(NDWI_THRESHOLD)

def get_NDSI_threshold(s2_image,NDSI_THRESHOLD):
    return s2_image.normalizedDifference(['B11','B3']).gt(NDSI_THRESHOLD)

def get_NDVI_NDWI_threshold(s2_image,NDVI_THRESHOLD,NDWI_THRESHOLD):
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD).multiply(s2_image.normalizedDifference(['B3','B8']).lt(NDWI_THRESHOLD))

def get_NDVI_ANDWI_threshold(s2_image,NDVI_THRESHOLD,NDWI_THRESHOLD):
    red = s2_image.select('B4')
    green = s2_image.select('B3')
    blue = s2_image.select('B2')
    nir = s2_image.select('B8')
    swir1 = s2_image.select('B11')
    swir2 = s2_image.select('B12')
    andwi = (red.add(green).add(blue).subtract(nir).subtract(swir1).subtract(swir2)).divide(red.add(green).add(blue).add(nir).add(swir1).add(swir2)).rename('ANDWI')
    return s2_image.normalizedDifference(['B8','B4']).lt(NDVI_THRESHOLD).multiply(andwi.lt(NDWI_THRESHOLD))

def clip_to_geometry(s2_image,geometry):
    return s2_image.clip(geometry)

def count_pixels(image,polygon):
    reduced = image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=polygon,
        scale=10,
        maxPixels=1e13,
        bestEffort=True,
        crs='EPSG:4326')
    return reduced

def add_cloud_bands(s2_image,cld_prb_thresh):
    cloud_probability = ee.Image(s2_image.get('s2cloudless')).select('probability')
    is_cloud = cloud_probability.gt(cld_prb_thresh).rename('clouds')
    return s2_image.addBands(ee.Image([cloud_probability, is_cloud]))

def add_shadow_bands(s2_image,nir_drk_thresh,cld_prj_dist,sr_band_scale):
    not_water = s2_image.select('SCL').neq(6) 
    # sr_band_scale = 1e4
    dark_pixels = s2_image.select('B8').lt(nir_drk_thresh*sr_band_scale).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(s2_image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    cld_proj = (s2_image.select('clouds').directionalDistanceTransform(shadow_azimuth, cld_prj_dist*10)
        .reproject(**{'crs': s2_image.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return s2_image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cloud_shadow_mask(s2_image,buffer_val,cld_prb_thresh,nir_drk_thresh,sr_band_scale,cld_prj_dist):
    img_cloud = add_cloud_bands(s2_image,cld_prb_thresh)
    img_cloud_shadow = add_shadow_bands(img_cloud,nir_drk_thresh,cld_prj_dist,sr_band_scale)
    #this works:
    is_cloud_shadow = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0).rename('cloudmask')
    #this doesn't work:
    # is_cloud_shadow = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    # is_cloud_shadow = (is_cloud_shadow.focalMin(2).focalMax(buffer_val*2/20)
    #     .reproject(**{'crs': s2_image.select([0]).projection(), 'scale': 20})
    #     .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cloud_shadow)

def apply_cloud_shadow_mask(s2_image):
    not_cloud_shadow = s2_image.select('cloudmask').Not()
    return s2_image.select('B.*').updateMask(not_cloud_shadow)

def csv_to_convex_hull_shp(df,csv_file,writing=True):
    lon = np.asarray(df.lon)
    lat = np.asarray(df.lat)
    date_list = df.date.unique().tolist()
    date_list_cleaned = [x for x in date_list if str(x) != 'nan']
    idx = [list(df.date).index(x) for x in set(list(df.date))]
    idx_sorted = np.sort(idx)
    idx_sorted = np.append(idx_sorted,len(df))
    gdf = gpd.GeoDataFrame()
    for i in range(len(date_list_cleaned)):
        date_str = date_list_cleaned[i]
        lonlat = np.column_stack((lon[idx_sorted[i]:idx_sorted[i+1]],lat[idx_sorted[i]:idx_sorted[i+1]]))
        if len(lonlat) == 1:
            icesat2_date_polygon = shapely.geometry.Point(lonlat)
        elif len(lonlat) == 2:
            icesat2_date_polygon = shapely.geometry.LineString(lonlat)
        else:
            icesat2_date_polygon = shapely.geometry.Polygon(lonlat)
        conv_hull = icesat2_date_polygon.convex_hull
        conv_hull_buffered = conv_hull.buffer(5E-4)
        df_tmp = pd.DataFrame({'date':[date_str]})
        gdf_tmp = gpd.GeoDataFrame(df_tmp,geometry=[conv_hull_buffered],crs='EPSG:4326')
        gdf = gpd.GeoDataFrame(pd.concat([gdf,gdf_tmp],ignore_index=True))
    gdf = gdf.set_crs('EPSG:4326')
    if writing == True:
        output_file = f'{os.path.splitext(csv_file)[0]}.shp'
        gdf.to_file(output_file)
    return gdf

def find_s2_image(date,geometry,s2,s2_cloud_probability,dt_search,cloud_filter,buffer,cld_prb_thresh,nir_drk_thresh,sr_band_scale,cld_prj_dist):
    csv_date_ee = ee.Date.parse('YYYY-MM-dd',date)
    csv_geometry = geometry
    csv_geometry_bounds = csv_geometry.bounds
    csv_geometry_xy = [[x,y] for x,y in zip(csv_geometry.exterior.xy[0],csv_geometry.exterior.xy[1])]
    polygon = ee.Geometry.Polygon(csv_geometry_xy)
    i_date = datetime.datetime.strptime(date,'%Y-%m-%d') - datetime.timedelta(days=dt_search)
    i_date = i_date.strftime('%Y-%m-%d')
    f_date = datetime.datetime.strptime(date,'%Y-%m-%d') + datetime.timedelta(days=dt_search+1) #because f_date is exclusive
    f_date = f_date.strftime('%Y-%m-%d')
    s2_date_region = s2.filterDate(i_date,f_date).filterBounds(polygon)
    s2_cloud_probability_date_region = s2_cloud_probability.filterDate(i_date,f_date).filterBounds(polygon)
    s2_merged_date_region = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary':s2_date_region.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)),
        'secondary':s2_cloud_probability_date_region,
        'condition':ee.Filter.equals(**{
            'leftField':'system:index',
            'rightField':'system:index'
        })
    }))
    ymd_ee = (s2_merged_date_region
        .map(lambda image : image.set('date', image.date().format("YYYYMMdd")))
        .distinct('date')
        .aggregate_array('date'))
    ymd_dates = ymd_ee.map(lambda s : ee.Date.parse('YYYYMMdd',s))
    ymd_length = ymd_dates.length().getInfo()
    if ymd_length == 0:
        return None,None,None
    dt_s2_images = ee.Array(ymd_dates.map(lambda s : ee.Date(s).difference(csv_date_ee,'day')))
    s2_subset = (ymd_ee.map(lambda date : s2_merged_date_region.filterMetadata('system:index','contains', date)))
    overlap_ratio = ee.Array(s2_subset.map(lambda img : ee.ImageCollection(img).geometry().intersection(polygon).area().divide(polygon.area())))
    filtered_clouds_single_date = s2_subset.map(lambda img : ee.ImageCollection(img).map(lambda img2 : img2.clip(polygon)))
    filtered_clouds_single_date = filtered_clouds_single_date.map(lambda img : ee.ImageCollection(img).map(lambda img2 : add_cloud_shadow_mask(img2,buffer,cld_prb_thresh,nir_drk_thresh,sr_band_scale,cld_prj_dist)))
    # cloudmask_single_date = filtered_clouds_single_date.map(lambda img : ee.ImageCollection(img).mosaic().select('cloudmask').selfMask())
    # notcloudmask_single_date = filtered_clouds_single_date.map(lambda img : ee.ImageCollection(img).mosaic().select('cloudmask').neq(1).selfMask())
    # n_clouds = ee.Array(cloudmask_single_date.map(lambda img : count_pixels(ee.Image(img),polygon).get('cloudmask')))
    # n_not_clouds = ee.Array(notcloudmask_single_date.map(lambda img : count_pixels(ee.Image(img),polygon).get('cloudmask')))
    #Temporary fix!
    cloudmask_single_day = filtered_clouds_single_date.map(lambda img : ee.ImageCollection(img).mosaic().select('clouds').eq(1).selfMask())
    notcloudmask_single_day = filtered_clouds_single_date.map(lambda img : ee.ImageCollection(img).mosaic().select('clouds').eq(0).selfMask())
    n_clouds = ee.Array(cloudmask_single_day.map(lambda img : count_pixels(ee.Image(img),polygon).get('clouds')))
    n_not_clouds = ee.Array(notcloudmask_single_day.map(lambda img : count_pixels(ee.Image(img),polygon).get('clouds')))
    cloud_percentage = n_clouds.divide(n_clouds.add(n_not_clouds))
    f1_score = (cloud_percentage.multiply(ee.Number(-1)).add(ee.Number(1))).multiply(overlap_ratio).divide((cloud_percentage.multiply(ee.Number(-1)).add(ee.Number(1))).add(overlap_ratio)).multiply(ee.Number(2))
    f1_modified = f1_score.subtract(dt_s2_images.abs().divide(ee.Number(100))).subtract(dt_s2_images.gt(ee.Number(0)).divide(ee.Number(200)))
    idx_select = f1_modified.argmax().get(0)
    ymd_select = ymd_ee.get(idx_select)
    ymd_select_info = ymd_select.getInfo()
    s2_select = ee.ImageCollection(filtered_clouds_single_date.get(idx_select))
    s2_select_clouds_removed = s2_select.map(lambda img : apply_cloud_shadow_mask(img))
    s2_select_clouds_removed_mosaic = s2_select_clouds_removed.mosaic()
    return s2_select_clouds_removed_mosaic,ymd_select_info,polygon

def export_to_drive(img,filename,geometry,loc_name):
    basename = os.path.splitext(filename)[0]
    export_task = ee.batch.Export.image.toDrive(image=img,
                                        description=basename,
                                        scale=10,
                                        region=geometry,
                                        fileNamePrefix=basename,
                                        crs='EPSG:4326',
                                        fileFormat='GeoTIFF',
                                        folder=f'GEE_{loc_name}')
    export_task.start()
    waiting = True
    while waiting:
        if export_task.status()['state'] == 'COMPLETED':
            waiting = False
        elif export_task.status()['state'] == 'FAILED':
            waiting = False
            return None
        else:
            time.sleep(5)
    return 0

def get_google_drive_credentials(token_json,credentials_json,SCOPES):
    creds = None
    if os.path.exists(token_json):
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file(token_json, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                credentials_json, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_json, 'w') as token:
            token.write(creds.to_json())
    return creds

def get_google_drive_dir_id(gdrive_service,dir_name):
    page_token = None
    folders = []
    query = f"name = '{dir_name}' and mimeType = 'application/vnd.google-apps.folder'"
    try:
        while True:
            response = gdrive_service.files().list(q=query,
                                                   spaces='drive',
                                                   fields='nextPageToken, '
                                                   'files(id, name)',
                                                   pageToken=page_token).execute()
            folders.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except googleapiclient.errors.HttpError as error:
        print(f'An error occurred: {error}')
        folders = None
    return folders

def get_google_drive_file_id(gdrive_service,dir_id,file_name):
    '''
    query = f"name = '{file_base}' and mimeType = 'image/tiff' and '{dir_id}' in parents"
    '''
    file_base = os.path.splitext(file_name)[0]
    page_token = None
    files = []
    query = f"mimeType = 'image/tiff' and '{dir_id}' in parents"
    try:
        while True:
            response = gdrive_service.files().list(q=query,
                                                   spaces='drive',
                                                   fields='nextPageToken, '
                                                   'files(id, name)',
                                                   pageToken=page_token).execute()
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except googleapiclient.errors.HttpError as error:
        print(f'An error occurred: {error}')
        files = None
    return files

def download_google_drive_id(gdrive_service,file_id):
    try:
        request = gdrive_service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = googleapiclient.http.MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    except googleapiclient.errors.HttpError as error:
        print(F'An error occurred: {error}')
        file = None
        return 0
    return file.getvalue()

def download_img_google_drive(filename,output_folder,tmp_dir,token_json,credentials_json,SCOPES):
    creds = get_google_drive_credentials(token_json,credentials_json,[SCOPES])
    service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)
    folder_list = get_google_drive_dir_id(service,output_folder)
    for folder in folder_list:
        folder_id = folder['id']
        folder_name = folder['name']
        if folder_name != output_folder:
            continue
        file_list = get_google_drive_file_id(service,folder_id,filename)
        idx_select = np.argwhere([f['name'] == filename for f in file_list])
        if len(idx_select) == 0:
            continue
        idx_select = np.atleast_1d(idx_select.squeeze())[0]
        file_id = file_list[idx_select]['id']
        download_code = download_google_drive_id(service,file_id)
        if download_code == 0:
            continue
        else:
            f = open(f'{tmp_dir}{filename}','wb')
            f.write(download_code)
            f.close()
            return 0
    return None

def get_idx_subset_date(t_full,t_select):
    t_date = np.asarray([t[:10] for t in t_full])
    idx = t_date == t_select
    return idx

def polygonize_tif(img):
    img_nodata = img.replace('.tif','_nodata_0.tif')
    shp = img.replace('.tif','.shp')
    nodata_command = f'gdal_translate -q -a_nodata 0 {img} {img_nodata}'
    polygonize_command = f'gdal_polygonize.py -q {img_nodata} -f "ESRI Shapefile" {shp}'
    subprocess.run(nodata_command,shell=True)
    subprocess.run(polygonize_command,shell=True)
    return shp

def parallel_s2_image(idx,date,geometry,loc_name,subset_file,gee_dict):
    t_start = datetime.datetime.now()
    print(f'Working on {idx}...')
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    s2_cloud_probability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    tmp_dir = gee_dict['tmp_dir']
    output_folder_gdrive = f'GEE_{loc_name}'
    landmask_c_file = gee_dict['landmask_c_file']

    dt_search = gee_dict['DT_SEARCH']
    cloud_filter = gee_dict['CLOUD_FILTER']
    cld_prb_thresh = gee_dict['CLD_PRB_THRESH']
    nir_drk_thresh = gee_dict['NIR_DRK_THRESH']
    cld_prj_dist = gee_dict['CLD_PRJ_DIST']
    buffer = gee_dict['BUFFER']
    sr_band_scale = gee_dict['SR_BAND_SCALE']
    ndvi_threshold = gee_dict['NDVI_THRESHOLD']
    ndwi_threshold = gee_dict['NDWI_THRESHOLD']
    scopes = gee_dict['SCOPES']
    token_json = gee_dict['token_json']
    credentials_json = gee_dict['credentials_json']

    i2_ymd = date.replace('-','')
    s2_image,s2_ymd,s2_geometry = find_s2_image(date,geometry,s2,s2_cloud_probability,dt_search,cloud_filter,buffer,cld_prb_thresh,nir_drk_thresh,sr_band_scale,cld_prj_dist)
    if s2_image is None:
        print(f'No suitable Sentinel-2 data for {idx}.')
        return 0
    ndvi_ndwi_threshold = get_NDVI_ANDWI_threshold(s2_image,ndvi_threshold,ndwi_threshold)
    ndvi_ndwi_threshold_filename = f'{loc_name}_ATL03_{i2_ymd}_S2_{s2_ymd}_NDVI_NDWI_threshold.tif'
    export_code = export_to_drive(ndvi_ndwi_threshold,ndvi_ndwi_threshold_filename,s2_geometry,loc_name)
    if export_code is None:
        print(f'Google Drive export failed for {idx}.')
        return 0
    t_end = datetime.datetime.now()
    dt = t_end - t_start
    print(f'Processing Sentinel-2 for {idx} took {dt.seconds + dt.microseconds/1e6:.1f} s.')

    t_start = datetime.datetime.now()
    download_code = download_img_google_drive(ndvi_ndwi_threshold_filename,output_folder_gdrive,tmp_dir,token_json,credentials_json,scopes)
    if download_code is None:
        print(f'Could not download image {idx} from Google Drive.')
        return 0
    
    ndvi_ndwi_threshold_local_file = f'{tmp_dir}{ndvi_ndwi_threshold_filename}'
    ndvi_ndwi_threshold_shp = polygonize_tif(ndvi_ndwi_threshold_local_file)
    gdf_ndvi_ndwi_threshold = gpd.read_file(ndvi_ndwi_threshold_shp)
    lon_ndvi_ndwi,lat_ndvi_ndwi = get_lonlat_gdf(gdf_ndvi_ndwi_threshold)
    df_subset_date = pd.read_csv(subset_file)
    lon_subset_date = np.asarray(df_subset_date.lon)
    lat_subset_date = np.asarray(df_subset_date.lat)

    landmask = landmask_csv(lon_subset_date,lat_subset_date,lon_ndvi_ndwi,lat_ndvi_ndwi,landmask_c_file,1)
    df_subset_date_masked = df_subset_date[landmask].reset_index(drop=True)

    output_file = f'{tmp_dir}{loc_name}_{i2_ymd}_Filtered_NDVI_NDWI.txt'
    df_subset_date_masked.to_csv(output_file,index=False,float_format='%.6f',header=None) #in this case header=None prevents multiple headers in full output file
    t_end = datetime.datetime.now()
    dt = t_end - t_start
    print(f'Applying filter for {idx} took {dt.seconds + dt.microseconds/1e6:.1f} s.')
    return 1

def main():
    ee.Initialize()
    warnings.simplefilter(action='ignore')
    config_file = 'utils_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to csv to filter')
    parser.add_argument('--machine',help='Machine name',default='t')
    parser.add_argument('--N_cpus',help='Number of CPUs to use',default=1,type=int)
    args = parser.parse_args()
    input_file = args.input_file
    machine_name = args.machine
    N_cpus = args.N_cpus

    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    if machine_name == 'b':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')

    DT_SEARCH = config.getint('GEE_CONSTANTS','DT_SEARCH')
    CLOUD_FILTER = config.getint('GEE_CONSTANTS','CLOUD_FILTER')
    CLD_PRB_THRESH = config.getint('GEE_CONSTANTS','CLD_PRB_THRESH')
    NIR_DRK_THRESH = config.getfloat('GEE_CONSTANTS','NIR_DRK_THRESH')
    CLD_PRJ_DIST = config.getint('GEE_CONSTANTS','CLD_PRJ_DIST')
    BUFFER = config.getint('GEE_CONSTANTS','BUFFER')
    SR_BAND_SCALE = config.getfloat('GEE_CONSTANTS','SR_BAND_SCALE')
    NDVI_THRESHOLD = config.getfloat('GEE_CONSTANTS','NDVI_THRESHOLD')
    NDWI_THRESHOLD = config.getfloat('GEE_CONSTANTS','NDWI_THRESHOLD')
    SCOPES = config.get('GENERAL_CONSTANTS','SCOPES')
    token_json = config.get('GDRIVE_PATHS','token_json')
    credentials_json = config.get('GDRIVE_PATHS','credentials_json')
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')

    gee_dict = {'DT_SEARCH':DT_SEARCH,'CLOUD_FILTER':CLOUD_FILTER,'CLD_PRB_THRESH':CLD_PRB_THRESH,
                'NIR_DRK_THRESH':NIR_DRK_THRESH,'CLD_PRJ_DIST':CLD_PRJ_DIST,'BUFFER':BUFFER,
                'SR_BAND_SCALE':SR_BAND_SCALE,
                'NDVI_THRESHOLD':NDVI_THRESHOLD,'NDWI_THRESHOLD':NDWI_THRESHOLD,
                'SCOPES':SCOPES,'token_json':token_json,'credentials_json':credentials_json,
                'landmask_c_file':landmask_c_file,'tmp_dir':tmp_dir
                }

    df_icesat2 = pd.read_csv(input_file)
    df_icesat2['date'] = np.asarray([t[:10] for t in df_icesat2.time])
    gdf_conv_hull = csv_to_convex_hull_shp(df_icesat2.copy(),input_file)

    loc_name = input_file.split('/')[-1].split('_ATL03')[0]
    gdf_conv_hull['loc_name'] = loc_name

    for date in gdf_conv_hull.date:
        df_icesat2[df_icesat2['date'] == date].to_csv(f'{tmp_dir}{loc_name}_{date.replace("-","")}_ATL03.txt',index=False,float_format='%.6f')

    index_array = gdf_conv_hull.index.to_numpy(dtype=int)
    date_array = np.asarray(gdf_conv_hull.date)
    geometry_array = np.asarray(gdf_conv_hull.geometry)
    loc_name_array = np.asarray(gdf_conv_hull.loc_name)
    subset_file_array = np.asarray(sorted(glob.glob(f'{tmp_dir}{loc_name}*_ATL03.txt')))

    t_start_full = datetime.datetime.now()
    ir = itertools.repeat

    p = multiprocessing.Pool(N_cpus)
    p.starmap(parallel_s2_image,zip(index_array,date_array,geometry_array,loc_name_array,subset_file_array,ir(gee_dict)))
    p.close()

    file_list = sorted(glob.glob(f'{tmp_dir}{loc_name}_*_Filtered_NDVI_NDWI.txt'))
    output_full_file = input_file.replace('.txt','_Filtered_NDVI_NDWI.txt')
    orig_header = ','.join(df_icesat2.columns.to_list())
    cat_command = f'cat {" ".join(file_list)} > {output_full_file}'
    sed_command = f"sed -i '1i {orig_header}' {output_full_file}"
    subprocess.run(cat_command,shell=True)
    subprocess.run(sed_command,shell=True)
    ndvi_ndwi_dir = f'{os.path.dirname(input_file)}/NDVI_NDWI/'
    if not os.path.isdir(ndvi_ndwi_dir):
        os.mkdir(ndvi_ndwi_dir)
    mv_command = f'mv {tmp_dir}{loc_name}* {ndvi_ndwi_dir}'
    subprocess.run(mv_command,shell=True)
    t_end_full = datetime.datetime.now()
    dt_full = t_end_full - t_start_full
    if dt_full.seconds > 3600:
        print(f'{loc_name} took {np.floor(dt_full.seconds/3600).astype(int)} hour(s), {np.floor(dt_full.seconds/60).astype(int)} minute(s), {np.mod(dt_full.seconds,60) + dt_full.microseconds/1e6:.1f} s.')
    else:
        print(f'{loc_name} took {np.floor(dt_full.seconds/60).astype(int)} minute(s), {np.mod(dt_full.seconds,60) + dt_full.microseconds/1e6:.1f} s.')

if __name__ == '__main__':
    main()