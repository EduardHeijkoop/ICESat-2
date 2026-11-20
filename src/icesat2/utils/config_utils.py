from dataclasses import dataclass
import configparser
import datetime

from icesat2.utils.general_utils import add_home

@dataclass
class ICESat2Config:
    #GENERAL_CONSTANTS
    earthdata_username: str
    #GENERAL_PATHS
    tmp_dir: str
    project_dir: str
    osm_shp_file: str
    #ICESAT2_OPTIONS
    N_threads: int
    landmask_flag: bool
    landmask_shp_buffer: float
    t_start: str
    t_end: str
    N_split_years: int
    #ATL03_OPTIONS
    beam_strength: str
    photon_confidence: str
    high_quality_flag: bool
    version: int
    row_group_size: int



def get_icesat2_config(config):
    icesat2_config = ICESat2Config(
        #GENERAL_CONSTANTS
        earthdata_username = config.get('GENERAL_CONSTANTS','earthdata_username',fallback=None),
        #GENERAL_PATHS
        tmp_dir = add_home(config.get('GENERAL_PATHS','tmp_dir',fallback='')),
        project_dir = add_home(config.get('GENERAL_PATHS','project_dir',fallback='')),
        osm_shp_file = add_home(config.get('GENERAL_PATHS','osm_shp_file',fallback='')),
        #ICESAT2_OPTIONS
        N_threads = config.getint('ICESAT2_OPTIONS','N_threads',fallback=1),
        landmask_flag = config.getboolean('ICESAT2_OPTIONS','landmask_flag',fallback=True),
        landmask_shp_buffer = config.getfloat('ICESAT2_OPTIONS','landmask_shp_buffer',fallback=0.01),
        t_start = replace_empty(config.get('ICESAT2_OPTIONS','t_start',fallback=''),fallback='2018-09-01'),
        t_end = replace_empty(config.get('ICESAT2_OPTIONS','t_end',fallback=''),fallback=datetime.datetime.now().strftime('%Y-%m-%d')),
        N_split_years = config.getint('ICESAT2_OPTIONS','N_split_years',fallback=1),
        #ATL03_OPTIONS
        beam_strength = config.get('ATL03_OPTIONS','beam_strength',fallback='strong'),
        photon_confidence = config.get('ATL03_OPTIONS','photon_confidence',fallback='high'),
        high_quality_flag = config.getboolean('ATL03_OPTIONS','high_quality_flag',fallback=True),
        version = config.getint('ATL03_OPTIONS','version',fallback=6),
        row_group_size = config.getint('ATL03_OPTIONS','row_group_size',fallback=1_000_000)
        )
    return icesat2_config

def optional(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ('', 'none', 'null'):
        return None
    return value

def replace_empty(value,fallback):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ('', 'none', 'null'):
        return fallback
    return value