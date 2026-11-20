# import harmony
import configparser
import os
import datetime
import numpy as np
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

# from icesat2.utils.general_utils import add_home,lonlat2epsg
from icesat2.utils.config_utils import get_icesat2_config


class ATL03Analyzer:
    def __init__(self,configuration: configparser.ConfigParser):
        self.configuration_data = configuration
        self.config = get_icesat2_config(configuration)

    def process(self,output_prefix,file_list):
        for granule in file_list:
            self._read_granule(output_prefix,granule)
        full_parquet_file = os.path.join(*[self.config.project_dir,output_prefix,f'{output_prefix}_ATL03.parquet'])
        return full_parquet_file
    
    def _read_granule(self,output_prefix,granule_path):
        beam_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']
        # print(os.path.basename(granule_path))
        with h5py.File(granule_path,'r') as f:
            # rgt = int(np.ravel(f['orbit_info/rgt'][:])[0])
            # cycle = int(np.ravel(f['orbit_info/cycle_number'][:])[0])
            # orbit = (cycle-1)*1387+rgt
            main_dir = os.path.join(*[self.config.project_dir,output_prefix,f'{output_prefix}_ATL03.parquet'])
            date = datetime.datetime.strptime(os.path.basename(granule_path).split('_')[1],'%Y%m%d%H%M%S').strftime('%Y-%m-%d')
            date_dir = os.path.join(*[main_dir,f'date={date}'])
            for beam in beam_list:
                if f'{beam}/heights/h_ph' not in f:
                    continue
                strength = self._get_beam_strength(f,beam)
                strength_dir = os.path.join(*[date_dir,f'strength={strength}'])
                beam_dir = os.path.join(*[strength_dir,f'beam={beam}'])
                # if strength is None:
                if strength is None or (strength != self.config.beam_strength and self.config.beam_strength != 'all'):
                    continue
                os.makedirs(beam_dir,exist_ok=True)
                beam_table = self._read_beam(f,beam,conf=self.config.photon_confidence,high_quality=self.config.high_quality_flag)
                if beam_table is None:
                    continue
                output_file = os.path.basename(granule_path).replace(f'_subsetted_{beam}.h5',f'_{beam}.parquet')
                full_output_file = os.path.join(*[beam_dir,output_file])
                tmp_output_file = os.path.join(*[beam_dir,f'.{output_file}.tmp'])
                #write to temporary file first, because if that fails or breaks, full parquet can still be read
                pq.write_table(beam_table, tmp_output_file, compression='zstd', row_group_size=self.config.row_group_size)
                os.replace(tmp_output_file, full_output_file)
                # if beam_table is not None:
                #     table_list.append(beam_table)
        # if len(table_list) == 0:
        #     return None
        # full_table = pa.concat_tables(table_list)
        # return full_table

    def _read_beam(self,f,beam,conf='high',high_quality=True):
        # strength = self._get_beam_strength(f,beam)
        # if strength is None or (strength != self.config.beam_strength and self.config.beam_strength != 'all'):
            #if strength is None (S/C is flying neither forwards or backwards) OR beam is wrong strength
            # return None
        schema = self._get_schema()
        h = f[f'{beam}/heights/h_ph'][:]
        N_ph = len(h)
        if N_ph == 0:
            return None
        rgt = int(np.ravel(f['orbit_info/rgt'][:])[0])
        cycle = int(np.ravel(f['orbit_info/cycle_number'][:])[0])
        seg_of_ph = self._photon_to_segment(f,beam,N_ph)
        signal_conf_land = f[f'{beam}/heights/signal_conf_ph'][:,0]
        pod_ppd_flag = f[f'/{beam}/geolocation/podppd_flag'][:]
        pod_ppd_flag = pod_ppd_flag[seg_of_ph]
        idx_pod_ppd = pod_ppd_flag == 0
        quality_ph = f[f'/{beam}/heights/quality_ph'][:]
        if conf == 'high':
            idx_conf = signal_conf_land == 4
        elif conf == 'medium':
            idx_conf = np.logical_or(signal_conf_land==3,signal_conf_land==4)
        else:
            idx_conf = np.ones(N_ph).astype(bool)
        if high_quality == True:
            idx_quality = quality_ph == 0
        else:
            idx_quality = np.ones(N_ph).astype(bool)
        idx_keep = np.all((idx_conf,idx_quality,idx_pod_ppd),axis=0)
        N_ph_keep = np.sum(idx_keep)
        seg_of_ph = seg_of_ph[idx_keep]
        #Full photon rate:
        h = h[idx_keep]
        lon = f[f'{beam}/heights/lon_ph'][idx_keep]
        lat = f[f'{beam}/heights/lat_ph'][idx_keep]
        sdp = np.ravel(f['ancillary_data/atlas_sdp_gps_epoch'][:])[0]
        delta_time = f[f'{beam}/heights/delta_time'][idx_keep]
        delta_time_total = delta_time + sdp
        timestamps = self._delta_time_to_unix(delta_time_total)
        #Reference photon rate:
        sigma_height = self._get_ref_photon_var(f,f'/{beam}/geolocation/sigma_h',seg_of_ph)
        tide_ocean = self._get_ref_photon_var(f,f'/{beam}/geophys_corr/tide_ocean',seg_of_ph)
        dac = self._get_ref_photon_var(f,f'/{beam}/geophys_corr/dac',seg_of_ph)
        tide_equilibrium = self._get_ref_photon_var(f,f'/{beam}/geophys_corr/tide_equilibrium',seg_of_ph)
        geoid = self._get_ref_photon_var(f,f'/{beam}/geophys_corr/geoid',seg_of_ph)
        geoid_free2mean = self._get_ref_photon_var(f,f'/{beam}/geophys_corr/geoid_free2mean',seg_of_ph)
        #Extra info:
        table = pa.table(
            {
                'time':timestamps.astype(np.int64),
                'lon':lon,
                'lat':lat,
                'height':h,
                'sigma_height':sigma_height,
                # 'beam':pa.DictionaryArray.from_arrays(pa.array(np.zeros(N_ph_keep, dtype=np.int8)), pa.array([beam])),
                # 'strength':pa.DictionaryArray.from_arrays(pa.array(np.zeros(N_ph_keep, dtype=np.int8)), pa.array([strength])),
                'rgt':pa.DictionaryArray.from_arrays(pa.array(np.zeros(N_ph_keep, dtype=np.int8)), pa.array([rgt])),
                'cycle':pa.DictionaryArray.from_arrays(pa.array(np.zeros(N_ph_keep, dtype=np.int8)), pa.array([cycle])),
                'geoid':geoid,
                'geoid_free2mean':geoid_free2mean,
                'tide_equilibrium':tide_equilibrium,
                'dac':dac,
                'tide_ocean':tide_ocean,
            },
            schema=schema
        )
        return table

    def _photon_to_segment(self,f,beam,n_ph):
        '''Index of the parent 20 m segment for every photon (-1 if unknown).'''
        cnt = f[f'{beam}/geolocation/segment_ph_cnt'][:].astype(np.int64)
        beg = f[f'{beam}/geolocation/ph_index_beg'][:].astype(np.int64)  # 1-based, 0 = no photons
        has = cnt > 0
        if not has.any():
            return np.full(n_ph, -1, dtype=np.int64)
        cnt_h, beg_h = cnt[has], beg[has]
        # make start indices relative, so it works whether or not a subsetter re-indexed them
        start = beg_h - beg_h.min()
        offsets = np.arange(cnt_h.sum()) - np.repeat(np.cumsum(cnt_h) - cnt_h, cnt_h)
        pos = np.repeat(start, cnt_h) + offsets
        seg = np.repeat(np.flatnonzero(has), cnt_h)
        ok = pos < n_ph
        seg_of_ph = np.full(n_ph, -1, dtype=np.int64)
        seg_of_ph[pos[ok]] = seg[ok]
        return seg_of_ph

    def _get_ref_photon_var(self,f,var,seg_of_ph):
        '''
        Returns variable that's originally at reference photon rate, at full photon rate
        '''
        d = f[var][:]
        d = d[seg_of_ph]
        return d

    def _get_beam_strength(self,f,beam):
        sc_orient = np.ravel(f['orbit_info/sc_orient'][:])[0]
        forward_dict = {
            'l':'weak',
            'r':'strong'
        }
        backward_dict = {
            'l':'strong',
            'r':'weak'
        }
        if sc_orient == 1:
            strength = forward_dict[beam[-1]]
        elif sc_orient == 0:
            strength = backward_dict[beam[-1]]
        else:
            return None
        return strength

    def _get_yaw_maneuvers(self):
        '''
        Up to date based on TechRefTable of 2026-11-09
        First maneuver (i.e. launch) on 2018-09-15 13:02:00 sets it to forward direction
        Each subsequent maneuver flips it (forward -> backward -> forward)
        '''
        yaw_maneuvers = [datetime.datetime(2018,9,15,13,2,0),
                        datetime.datetime(2018,12,28,18,53,8),
                        datetime.datetime(2019,9,7,1,4,6),
                        datetime.datetime(2020,5,14,1,49,3),
                        datetime.datetime(2021,1,15,15,17,1),
                        datetime.datetime(2021,10,2,2,20,1),
                        datetime.datetime(2022,6,9,1,31,19),
                        datetime.datetime(2023,2,9,16,59,14),
                        datetime.datetime(2023,10,27,13,26,50),
                        datetime.datetime(2024,6,25,9,45,16),
                        datetime.datetime(2025,3,3,14,47,4),
                        datetime.datetime(2025,11,20,0,57,40),
                        datetime.datetime(2026,7,21,14,8,40)]
        return yaw_maneuvers


    def _get_schema(self):
        schema = pa.schema([
            ('time', pa.timestamp('ns')),
            ('lon', pa.float64()),  # keep float64: float32 is ~1 m precision in degrees
            ('lat', pa.float64()),
            ('height', pa.float32()),  # float32 is still sub-mm for Earth heights
            ('sigma_height', pa.float32()),
            # ('beam', pa.dictionary(pa.int8(), pa.string())),
            # ('strength', pa.dictionary(pa.int8(), pa.string())),
            ('rgt', pa.dictionary(pa.int16(),pa.int64())),
            ('cycle', pa.dictionary(pa.int16(),pa.int64())),
            ('geoid', pa.float32()),
            ('geoid_free2mean', pa.float32()),
            ('tide_equilibrium', pa.float32()),
            ('dac', pa.float32()),
            ('tide_ocean', pa.float32()),
        ])
        return schema

    def _const_dict(self,value, n):
        '''Dictionary-encoded column holding one repeated string (costs ~nothing on disk).'''
        return pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, dtype=np.int8)), pa.array([value]))

    def _delta_time_to_unix(self,delta_time):
        '''
        ICESat-2 delta time is seconds since SDP (standard data product), which itself is seconds since GPS time of 1980-01-06 00:00:00
        Parquet wants to store as int64 seconds since 1970-01-01 00:00:00 UTC
        So we need leap seconds too, as of 2026-11-18 the last leap second was in 2017 before ICESat-2 launched
        np.datetime64 converts to int64 seconds since 1970
        '''
        leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
        t0_icesat2 = np.datetime64('1980-01-06T00:00:00','ns').astype(np.int64)
        # t0_unix = np.datetime64('1970-01-01T00:00:00','ns').astype(np.int64)
        timestamps_ns = np.rint((delta_time + leap_seconds) * 1e9).astype(np.int64)
        timestamps_ns += t0_icesat2
        timestamps_ns = timestamps_ns.view('datetime64[ns]')
        return timestamps_ns