import argparse
from osgeo import gdal,gdalconst,osr
import warnings
import configparser
import shapely
import sys

from icesat2.utils.config_utils import get_icesat2_config
from icesat2.utils.general_utils import add_home,get_raster_extents
from icesat2.utils.icesat2_utils import ICESat2Downloader
from icesat2.utils.atl03_utils import ATL03Analyzer
from icesat2.utils.duckdb_utils import DuckDBMask

def main(argv):
    args = parse_arguments(argv)
    extents = [float(f) for f in args.extents[0].split(',')] if len(args.extents) == 1 else [float(f.replace(',','')) for f in args.extents]
    topography_file = args.topography
    output_prefix = args.output_prefix
    product = args.product
    verbose = args.verbose
    surface_type = args.surface_type

    icesat2_configuration = configparser.ConfigParser(allow_no_value=True,interpolation=configparser.ExtendedInterpolation())
    icesat2_configuration.read(args.config)
    config = get_icesat2_config(icesat2_configuration)

    if topography_file is not None:
        lon_min,lon_max,lat_min,lat_max = get_raster_extents(topography_file,global_local_flag='global')
        bbox = shapely.geometry.box(lon_min,lat_min,lon_max,lat_max)
    elif extents is not None:
        bbox = shapely.geometry.box(*extents)

    file_list = ICESat2Downloader(icesat2_configuration).download(bbox,output_prefix,product,verbose=verbose)
    output_parquet = ATL03Analyzer(icesat2_configuration).process(output_prefix,file_list,surface_type=surface_type)
    print(f'Consolidated {len(file_list)} granules into {output_parquet}') if verbose else None
    if config.landmask_flag == True:
        keep = 'outside' if surface_type == 'ocean' else 'inside'
        output_parquet_masked = DuckDBMask(icesat2_configuration).mask(input_parquet_file=output_parquet,mask_file=config.osm_shp_file,keep=keep,verbose=verbose)
        print(f'Masked file: {output_parquet_masked}') if verbose else None

    '''
    If you want to load the data
    import polars as pl
    import os
    lf = pl.scan_parquet(os.path.join(*[config.project_dir,output_prefix,f'{output_prefix}_ATL03.parquet/**/*.parquet']), hive_partitioning=True,hive_schema={'date': pl.Date, 'strength': pl.String, 'beam': pl.String})
    #optional subsetting, use df = lf.collect() if you don't need it
    df = lf.filter((pl.col('strength') == 'strong') & (pl.col('date') >= pl.date(2025, 2, 1))).collect()
    '''

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Find flooding extents.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--topography', help='Path to input topography file with.')
    input_group.add_argument('--extents', help='Input extents: lon_min lat_min lon_max lat_max', nargs='*')
    parser.add_argument('--verbose',help='Verbose? Print extra info.',action='store_true',default=False)
    parser.add_argument('--surface_type',help='Surface type?',default='land',choices=['land','ocean'])
    parser.add_argument('--output_prefix', help='Output prefix for generated data (used with --extents).', default=None)
    parser.add_argument('--product', help='Which ICESat-2 product to use.', default='ATL03',choices=['ATL03','ATL08'])
    parser.add_argument('--config',help='Which config file to use?',type=str,default='src/icesat2/config/icesat2_config.ini')
    return parser.parse_args(args)
    

if __name__ == '__main__':
    gdal.DontUseExceptions()
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*Zarr format 3 specification.*')
    warnings.filterwarnings('ignore', message='.*grpcio < 1.83.0 does not support Post-Quantum Cryptography.*')
    warnings.filterwarnings('ignore', message='.*Calling sync() from within a running.*')

    main(sys.argv[1:])