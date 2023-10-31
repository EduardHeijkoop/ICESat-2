import numpy as np
import h5py
import subprocess
import os
from osgeo import gdal,gdalconst,osr
import pandas as pd

def analyze_icesat2_land(icesat2_dir,city_name,shp_data,beam_flag=False,weak_flag=False,sigma_flag=True,weight_flag=False):
    '''
    Given a directory of downloaded ATL03 hdf5 files,
    reads them and writes the high confidence photons to a CSV as:
    longitude,latitude,height [WGS84],time [UTC](,beam)
    '''
    icesat2_list = icesat2_dir+city_name + '/icesat2_list.txt'
    with open(icesat2_list) as f3:
        file_list = f3.read().splitlines()
    beam_list_r = ['gt1r','gt2r','gt3r']
    beam_list_l = ['gt1l','gt2l','gt3l']
    if weak_flag == True:
        beam_list_r = ['gt1l','gt2l','gt3l']
        beam_list_l = ['gt1r','gt2r','gt3r']
    lon_high_conf = np.empty([0,1],dtype=float) #Initialize arrays and start reading .h5 files
    lat_high_conf = np.empty([0,1],dtype=float)
    h_high_conf = np.empty([0,1],dtype=float)
    delta_time_total_high_conf = np.empty([0,1],dtype=float)
    if beam_flag == True:
        beam_high_conf = np.empty([0,1],dtype=str)
    if sigma_flag == True:
        sigma_h_high_conf = np.empty([0,1],dtype=float)
    for h5_file in file_list:
        full_file = icesat2_dir + city_name + '/' + h5_file
        atl03_data = h5py.File(full_file,'r')
        list(atl03_data.keys())
        sc_orient = atl03_data['/orbit_info/sc_orient'][0] #Select strong beams according to S/C orientation
        if sc_orient == 1:
            beam_list_req = beam_list_r
        elif sc_orient == 0:
            beam_list_req = beam_list_l
        elif sc_orient == 2:
            continue
        for beam in beam_list_req:
            '''
            Some beams don't actually have any height data in them, so this is done to skip those
            Sometimes only one or two beams are present, this also prevents looking for those
            '''
            heights_check = False
            heights_check = f'/{beam}/heights' in atl03_data
            if heights_check == False:
                continue
            tmp_lon = np.asarray(atl03_data[f'/{beam}/heights/lon_ph']).squeeze()
            tmp_lat = np.asarray(atl03_data[f'/{beam}/heights/lat_ph']).squeeze()
            tmp_h = np.asarray(atl03_data[f'/{beam}/heights/h_ph']).squeeze()
            tmp_sdp = np.asarray(atl03_data['/ancillary_data/atlas_sdp_gps_epoch']).squeeze()
            tmp_delta_time = np.asarray(atl03_data[f'/{beam}/heights/delta_time']).squeeze()
            tmp_delta_time_total = tmp_sdp + tmp_delta_time
            tmp_signal_conf = np.asarray(atl03_data[f'/{beam}/heights/signal_conf_ph'])
            tmp_high_conf = tmp_signal_conf[:,0] == 4
            tmp_quality = np.asarray(atl03_data[f'/{beam}/heights/quality_ph'])
            tmp_high_quality = tmp_quality == 0
            if weight_flag == True:
                tmp_weight = np.asarray(atl03_data[f'/{beam}/heights/weight_ph'])/255.0
                tmp_high_weight = tmp_weight > 0.8
            else:
                tmp_high_weight = np.ones(tmp_lon.shape,dtype=bool)
            if len(tmp_high_conf) < 100: #If fewer than 100 high confidence photons are in an hdf5 file, skip
                continue
            tmp_ph_index_beg = np.asarray(atl03_data[f'/{beam}/geolocation/ph_index_beg']).squeeze()
            tmp_ph_index_beg = tmp_ph_index_beg - 1
            tmp_segment_ph_cnt = np.asarray(atl03_data[f'/{beam}/geolocation/segment_ph_cnt']).squeeze()
            tmp_ref_ph_index = np.asarray(atl03_data[f'/{beam}/geolocation/reference_photon_index']).squeeze()
            tmp_ph_index_end = tmp_ph_index_beg + tmp_segment_ph_cnt
            tmp_podppd_flag = np.asarray(atl03_data[f'/{beam}/geolocation/podppd_flag']).squeeze()
            idx_ref_ph = tmp_segment_ph_cnt>0 #no valid photons to "create" a ref photon -> revert back to reference ground track, which we don't want, so select segments with >0 photons
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
            idx_flags = np.all((tmp_high_conf,tmp_high_quality,tmp_high_weight,tmp_idx_podppd,~idx_nan),axis=0)
            tmp_lon_high_conf = tmp_lon[idx_flags]
            tmp_lat_high_conf = tmp_lat[idx_flags]
            tmp_h_high_conf = tmp_h[idx_flags]
            tmp_delta_time_total_high_conf = tmp_delta_time_total[idx_flags]
            lon_high_conf = np.append(lon_high_conf,tmp_lon_high_conf)
            lat_high_conf = np.append(lat_high_conf,tmp_lat_high_conf)
            h_high_conf = np.append(h_high_conf,tmp_h_high_conf)
            delta_time_total_high_conf = np.append(delta_time_total_high_conf,tmp_delta_time_total_high_conf)
            if beam_flag == True:
                tmp_beam_high_conf = np.repeat(beam,len(tmp_lon_high_conf))
                beam_high_conf = np.append(beam_high_conf,tmp_beam_high_conf)
            if sigma_flag == True:
                tmp_sigma_h = np.asarray(atl03_data[f'/{beam}/geolocation/sigma_h']).squeeze()
                tmp_sigma_h = tmp_sigma_h[idx_ref_ph]
                tmp_sigma_h_full_ph = np.zeros(tmp_lon.shape)
                for i in range(len(tmp_ph_index_beg)):
                    tmp_sigma_h_full_ph[tmp_ph_index_beg[i]:tmp_ph_index_end[i]] = tmp_sigma_h[i]
                tmp_sigma_h_high_conf = tmp_sigma_h_full_ph[idx_flags]
                sigma_h_high_conf = np.append(sigma_h_high_conf,tmp_sigma_h_high_conf)

    '''
    A lot of data will be captured off the coast that we don't want,
    this is a quick way of getting rid of that
    Also prevents areas with no SRTM from being queried
    '''
    idx_lon = np.logical_or(lon_high_conf < np.min(shp_data.bounds.minx),lon_high_conf > np.max(shp_data.bounds.maxx))
    idx_lat = np.logical_or(lat_high_conf < np.min(shp_data.bounds.miny),lat_high_conf > np.max(shp_data.bounds.maxy))
    idx_tot = np.logical_or(idx_lon,idx_lat)
    lon_high_conf = lon_high_conf[~idx_tot]
    lat_high_conf = lat_high_conf[~idx_tot]
    h_high_conf = h_high_conf[~idx_tot]
    delta_time_total_high_conf = delta_time_total_high_conf[~idx_tot]
    if beam_flag == True:
        beam_high_conf = beam_high_conf[~idx_tot]
    else:
        beam_high_conf = None
    if sigma_flag == True:
        sigma_h_high_conf = sigma_h_high_conf[~idx_tot]
    else:
        sigma_h_high_conf = None

    return lon_high_conf,lat_high_conf,h_high_conf,delta_time_total_high_conf,beam_high_conf,sigma_h_high_conf

def copernicus_filter_icesat2(lon,lat,icesat2_file,icesat2_dir,city_name,copernicus_threshold,egm2008_file,buffer=0.01,keep_files_flag=False):
    lon_min = np.nanmin(lon) - buffer
    lon_max = np.nanmax(lon) + buffer
    lat_min = np.nanmin(lat) - buffer
    lat_max = np.nanmax(lat) + buffer
    output_dir = f'{icesat2_dir}{city_name}/'
    copernicus_wgs84_file = f'{output_dir}{city_name}_Copernicus_WGS84.tif'
    download_copernicus(lon_min,lon_max,lat_min,lat_max,egm2008_file,output_dir,copernicus_wgs84_file)
    copernicus_sampled_file = f'{output_dir}{city_name}_sampled_Copernicus.txt'
    print('Sampling Copernicus...')
    sample_raster(copernicus_wgs84_file, icesat2_file, copernicus_sampled_file,header='height_copernicus')
    print('Sampled Copernicus.')
    df_copernicus = pd.read_csv(copernicus_sampled_file)
    df_copernicus['dh'] = df_copernicus.height_copernicus - df_copernicus.height_icesat2
    copernicus_cond = np.asarray(np.abs(df_copernicus.dh) < copernicus_threshold)
    if keep_files_flag == False:
        subprocess.run(copernicus_wgs84_file,shell=True)
        subprocess.run(copernicus_sampled_file,shell=True)
    return copernicus_cond

def get_copernicus_tiles(lon_min,lon_max,lat_min,lat_max):
    COPERNICUS_list = []
    lon_range = range(int(np.floor(lon_min)),int(np.floor(lon_max))+1)
    lat_range = range(int(np.floor(lat_min)),int(np.floor(lat_max))+1)
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
            COPERNICUS_id = f'Copernicus_DSM_COG_10_{latLetter}{latCode}_00_{lonLetter}{lonCode}_00_DEM/Copernicus_DSM_COG_10_{latLetter}{latCode}_00_{lonLetter}{lonCode}_00_DEM.tif'
            COPERNICUS_list.append(COPERNICUS_id)
    return sorted(COPERNICUS_list)

def download_copernicus(lon_min,lon_max,lat_min,lat_max,egm2008_file,tmp_dir,output_file):
    tile_array = get_copernicus_tiles(lon_min,lon_max,lat_min,lat_max)
    copernicus_aws_base = 's3://copernicus-dem-30m/'
    merge_command = f'gdal_merge.py -q -o tmp_merged.tif '
    for tile in tile_array:
        dl_command = f'aws s3 cp --quiet --no-sign-request {copernicus_aws_base}{tile} .'
        subprocess.run(dl_command,shell=True,cwd=tmp_dir)
        if os.path.isfile(f'{tmp_dir}{tile.split("/")[-1]}'):
            merge_command = f'{merge_command} {tmp_dir}{tile.split("/")[-1]} '
    subprocess.run(merge_command,shell=True,cwd=tmp_dir)
    [subprocess.run(f'rm {tile.split("/")[-1]}',shell=True,cwd=tmp_dir) for tile in tile_array if os.path.isfile(f'{tmp_dir}{tile.split("/")[-1]}')]
    warp_command = f'gdalwarp -q -te {lon_min} {lat_min} {lon_max} {lat_max} tmp_merged.tif tmp_merged_clipped.tif'
    subprocess.run(warp_command,shell=True,cwd=tmp_dir)
    subprocess.run(f'rm tmp_merged.tif',shell=True,cwd=tmp_dir)
    if egm2008_file is not None:
        resample_raster(egm2008_file,f'{tmp_dir}tmp_merged_clipped.tif',f'{tmp_dir}EGM2008_resampled.tif',quiet_flag=True)
        calc_command = f'gdal_calc.py -A tmp_merged_clipped.tif -B EGM2008_resampled.tif --outfile={output_file} --calc=\"A+B\" --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
        subprocess.run(calc_command,shell=True,cwd=tmp_dir)
        subprocess.run(f'rm tmp_merged_clipped.tif EGM2008_resampled.tif',shell=True,cwd=tmp_dir)
    else:
        subprocess.run(f'mv tmp_merged_clipped.tif {output_file}',shell=True,cwd=tmp_dir)

def resample_raster(src_filename,match_filename,dst_filename,nodata=-9999,resample_method='bilinear',compress=True,quiet_flag=False):
    '''
    src = what you want to resample
    match = resample to this one's resolution
    dst = output
    method = nearest neighbor, bilinear (default), cubic, cubic spline
    '''
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue('AUTHORITY',1)
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename,wide,high,1,gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)
    if resample_method == 'nearest':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_NearestNeighbour)
    elif resample_method == 'bilinear':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_Bilinear)
    elif resample_method == 'cubic':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_Cubic)
    elif resample_method == 'cubicspline':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_CubicSpline)
    del dst # Flush
    if compress == True:
        compress_raster(dst_filename,nodata,quiet_flag)
    return None

def sample_raster(raster_path, csv_path, output_file,nodata='-9999',header=None,proj='wgs84'):
    output_dir = os.path.dirname(output_file)
    raster_base = os.path.splitext(raster_path.split('/')[-1])[0]
    if header is not None:
        cat_command = f"tail -n+2 {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{proj} {raster_path} > tmp_{raster_base}.txt"
    else:
        cat_command = f"cat {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{proj} {raster_path} > tmp_{raster_base}.txt"
    subprocess.run(cat_command,shell=True,cwd=output_dir)
    fill_nan_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp_{raster_base}.txt > tmp2_{raster_base}.txt"
    subprocess.run(fill_nan_command,shell=True,cwd=output_dir)
    if header is not None:
        subprocess.run(f"sed -i '1i {header}' tmp2_{raster_base}.txt",shell=True,cwd=output_dir)
    paste_command = f"paste -d , {csv_path} tmp2_{raster_base}.txt > {output_file}"
    subprocess.run(paste_command,shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/{nodata}/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"rm tmp_{raster_base}.txt tmp2_{raster_base}.txt",shell=True,cwd=output_dir)
    return None

def compress_raster(filename,nodata=-9999,quiet_flag = False):
    '''
    Compress a raster using gdal_translate
    '''
    file_ext = os.path.splitext(filename)[-1]
    tmp_filename = filename.replace(file_ext,f'_LZW{file_ext}')
    if nodata is not None:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -a_nodata {nodata} {filename} {tmp_filename}'
    else:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {filename} {tmp_filename}'
    if quiet_flag == True:
        compress_command = compress_command.replace('gdal_translate','gdal_translate -q')
    move_command = f'mv {tmp_filename} {filename}'
    subprocess.run(compress_command,shell=True)
    subprocess.run(move_command,shell=True)
    return None
