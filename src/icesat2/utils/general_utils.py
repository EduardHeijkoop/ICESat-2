import os
import numpy as np
from osgeo import gdal,gdalconst,osr
import subprocess

def add_home(file_path):
    '''
    Adds home directory to starting tilde in file paths
    Using ~ instead of home directory can cause issues
    But ~ can mask username and make common file paths usable across machines with the same .ini file
    '''
    if file_path.startswith('~'):
        home_dir = os.path.expanduser('~')
        file_path = os.path.join(*[home_dir,'/'.join(file_path.split('/')[1:])])
    return file_path

def get_extent(gt,cols,rows):
    '''
    Return list of corner coordinates from a geotransform
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    '''
    Reproject a list of x,y coordinates.
    x and y are in src coordinates, going to tgt
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def get_raster_extents(raster,global_local_flag='global'):
    '''
    Get global or local extents of a raster
    '''
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    local_ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # Apparently tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) might help
    if src_srs.GetAttrValue('AUTHORITY',1) == '4326':
        global_ext = local_ext
    else:
        global_ext = reproject_coords(local_ext,src_srs,tgt_srs)
    x_local = [item[0] for item in local_ext]
    y_local = [item[1] for item in local_ext]
    x_min_local = np.nanmin(x_local)
    x_max_local = np.nanmax(x_local)
    y_min_local = np.nanmin(y_local)
    y_max_local = np.nanmax(y_local)
    x_global = [item[0] for item in global_ext]
    y_global = [item[1] for item in global_ext]
    x_min_global = np.nanmin(x_global)
    x_max_global = np.nanmax(x_global)
    y_min_global = np.nanmin(y_global)
    y_max_global = np.nanmax(y_global)
    if global_local_flag.lower() == 'global':
        return x_min_global,x_max_global,y_min_global,y_max_global
    elif global_local_flag.lower() == 'local':
        return x_min_local,x_max_local,y_min_local,y_max_local
    else:
        return None
