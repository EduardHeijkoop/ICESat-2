import harmony
import configparser
import os
import glob
import datetime
import getpass
import warnings
import contextlib
import io
import pandas as pd
import numpy as np

from icesat2.utils.config_utils import get_icesat2_config


class ICESat2Downloader:
    def __init__(self,configuration: configparser.ConfigParser):
        self.configuration_data = configuration
        self.config = get_icesat2_config(configuration)

    def download(self,bbox,output_prefix,product,verbose=True):
        '''
        Uses NASA's Harmony API to subset and download ICESat-2 granules
        Jobs are submitted per time group (e.g. 2 years) and per beam, because the variable subsetting for the whole granule (x6) is too long for the GET request
        '''
        project_dir = self.config.project_dir
        output_dir = os.path.join(project_dir,output_prefix)
        os.makedirs(output_dir,exist_ok=True)
        beam_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']
        N_already_present = np.sum([len(glob.glob(os.path.join(*[output_dir,f'{product}_Data',beam,'*.h5']))) for beam in beam_list])
        if N_already_present > 0:
            input_val = input(f'Found {N_already_present} {product} granules. Use these? y/n?\n').lower()
            if input_val == 'y':
                return sorted(glob.glob(os.path.join(*[output_dir,f'{product}_Data','**','*.h5'])))
        pw = getpass.getpass('Enter your NASA EarthData password:')
        client = harmony.Client(auth=(self.config.earthdata_username,pw))
        cmr_id = self._version_to_cmr(product,self.config.version)
        capabilities_request = harmony.CapabilitiesRequest(collection_id=cmr_id)
        capabilities = client.submit(capabilities_request)
        if verbose == True:
            self._check_capabilities(capabilities)
        t_start_arr,t_end_arr = self._split_years()
        jobs = []
        jobs_beam_dict = {}
        full_file_list = []
        for t_start,t_end in zip(t_start_arr,t_end_arr):
            for beam in beam_list:
                beam_dir = os.path.join(*[output_dir,f'{product}_Data',beam])
                os.makedirs(beam_dir,exist_ok=True)
                var_list = self._get_variables(product,capabilities,beam)
                download_request = harmony.Request(
                    collection = harmony.Collection(id=cmr_id),
                    spatial = harmony.BBox(*bbox.bounds),
                    # temporal = {'start':datetime.datetime.strptime(self.config.t_start,'%Y-%m-%d'),
                    #             'stop':datetime.datetime.strptime(self.config.t_end,'%Y-%m-%d')},
                    temporal = {'start':t_start,
                                'stop':t_end},
                    variables = var_list
                )
                request_is_valid = download_request.is_valid()
                if request_is_valid == False:
                    raise ValueError('Invalid download request!')
                job_id = client.submit(download_request)
                jobs_beam_dict[job_id] = {'beam':beam,'t_start':t_start,'t_end':t_end}
                jobs.append(job_id)
        # print(f'Job ID: {job_id}')
        for job in jobs:
            print(f'{jobs_beam_dict[job]["t_start"]} - {jobs_beam_dict[job]["t_end"]} ({jobs_beam_dict[job]["beam"]})') if verbose else None
            client.wait_for_processing(job, show_progress=verbose)
            if verbose == True:
                job_summary = client.result_json(job)
                self._print_results(job_summary)
            beam = jobs_beam_dict[job]['beam']
            beam_dir = os.path.join(*[output_dir,f'{product}_Data',beam])
            with contextlib.redirect_stdout(io.StringIO()):
                futures = client.download_all(job, directory=beam_dir, overwrite=True)
                file_list = [f.result() for f in futures]  #Get filepaths, the .result() method also blocks further code until download is complete
                file_list = self._clean_file_list(product,file_list,beam)
                full_file_list += file_list
        return full_file_list

    def _get_variables(self,product,capabilities,beam):
        if product == 'ATL03':
            var_list = self._get_atl03_variables(capabilities,beam)
        elif product == 'ATL08':
            var_list = self._get_atl08_variables(capabilities,beam)
        else:
            raise NotImplementedError(f'Product {product} not supported (yet)!')
        return var_list

    def _get_atl03_variables(self,capabilities,beam):
        available_vars = [var['name'] for var in capabilities.get('variables',[])]
        var_list = []
        # beam_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']
        heights_vars = ['lon_ph','lat_ph','h_ph','delta_time','signal_conf_ph','quality_ph','weight_ph']
        geolocation_vars = ['reference_photon_lon','reference_photon_lat','ph_index_beg','segment_ph_cnt','reference_photon_index','sigma_h','sigma_lon','sigma_lat','podppd_flag']
        geophys_corr_vars = ['delta_time','tide_ocean','dac','tide_equilibrium','geoid','geoid_free2mean']
        other_vars = ['orbit_info/sc_orient','orbit_info/sc_orient_time','ancillary_data/atlas_sdp_gps_epoch','orbit_info/rgt','orbit_info/cycle_number']
        var_list += [f'{beam}/heights/{var}' for var in heights_vars]
        var_list += [f'{beam}/geolocation/{var}' for var in geolocation_vars]
        var_list += [f'{beam}/geophys_corr/{var}' for var in geophys_corr_vars]
        var_list += other_vars
        var_list_filtered = [var for var in var_list if var in available_vars]
        return var_list_filtered

    def _get_atl08_variables(self,capabilities):
        available_vars = [var['name'] for var in capabilities.get('variables',[])]
        var_list = []
        beam_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']

        var_list_filtered = [var for var in var_list if var in available_vars]
        return var_list_filtered

    def _clean_file_list(self,product,file_list,beam):
        new_file_list = []
        for f in file_list:
            new_name = os.path.join(os.path.dirname(f),f'{product}_{os.path.basename(f).split(f"{product}_")[1].replace(".h5",f"_{beam}.h5")}')
            os.rename(f,new_name)
            new_file_list.append(new_name)
        new_file_list.sort()
        return new_file_list

    def _check_capabilities(self,capabilities):
        try:
            subsetting_dict = capabilities['services'][0]['capabilities']['subsetting']
            print('Subsetting capabilities:')
            for key in subsetting_dict:
                print(f'{key} : {subsetting_dict[key]}')
        except KeyError:
            print('Cannot find subset capabilities! Still attempting to subset.')

    def _print_results(self,job_summary):
        '''
        Prints results of Harmony job
        '''
        job_status = job_summary['status']
        if job_status.lower() != 'successful':
            raise RuntimeError('Job did not complete successfully!')
        original_size = job_summary.get('originalDataSize', 'NaN')
        output_size = job_summary.get('outputDataSize','NaN')
        pct_reduction = job_summary.get('dataSizePercentChange','NaN')
        print(f'Achieved a {pct_reduction} ({original_size} -> {output_size}).')

    def _version_to_cmr(self,product,version):
        '''
        Converts ATL03 release/version number to CMR ID, which harmony can retrieve
        As of 2025-09-03 version 7 is the latest, but issues remain. NSIDC:
        "Isolated portions of ICESat-2 Release 7 data are affected by 1) gaps in
        instrument pointing estimates and 2) degraded orbit/pointing solutions around
        spacecraft maneuvers. These issues can cause large errors in photon geolocation
        and heights. Please refer to the ICESat-2 Technical References Table for a list
        of impacted dates/times and granules, and use caution when using these data.
        Note that the issues are being addressed, and the degraded files will be replaced."
        '''
        cmr_dict = {
            'atl03': {
                '6':{
                    'cmr_id':'C2596864127-NSIDC_CPRD',
                    'deprecated':True
                },
                '7':{
                    'cmr_id':'C3326974349-NSIDC_CPRD',
                    'deprecated':False
                }
            },
            'atl08': {
                '6':{
                    'cmr_id':'C2613553260-NSIDC_CPRD',
                    'deprecated':True
                },
                '7':{
                    'cmr_id':'C3565574177-NSIDC_CPRD',
                    'deprecated':False
                }
            },
        }
        cmr = cmr_dict[product.lower()][str(version)]['cmr_id']
        deprecated = cmr_dict[product.lower()][str(version)]['deprecated']
        if deprecated:
            warnings.warn(f'{product} v{version:03d} is deprecated! Unlikely you will get good data!')
        return cmr

    def _split_years(self):
        '''
        Splits input time into N number of years to distribute request over multiple jobs at NSIDC
        '''
        N_split_years = self.config.N_split_years
        t_start_datetime = datetime.datetime.strptime(self.config.t_start,'%Y-%m-%d')
        t_end_datetime = datetime.datetime.strptime(self.config.t_end,'%Y-%m-%d')

        t_start_arr = pd.date_range(datetime.datetime(t_start_datetime.year,1,1),t_end_datetime,freq=f'{N_split_years}YS').to_pydatetime()
        t_end_arr = np.asarray([datetime.datetime(t.year+N_split_years-1,12,31) for t in t_start_arr])

        t_start_arr[0] = t_start_datetime
        t_end_arr[-1] = t_end_datetime
        return t_start_arr,t_end_arr