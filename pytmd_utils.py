import numpy as np
import datetime

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.read_FES_model import extract_FES_constants


def ocean_tide_replacement(lon,lat,utc_time,config):
    #Given a set of lon,lat and utc time, computes FES2014 tidal elevations
    #pyTMD boilerplate
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model_dir = config.get('OCEAN_PATHS','model_dir')
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    time_datetime = np.asarray(list(map(datetime.datetime.fromisoformat,utc_time)))
    unique_date_list = np.unique([a.date() for a in time_datetime])
    tide_heights = np.empty(len(lon),dtype=np.float32)
    for unique_date in unique_date_list:
        idx_unique_date = np.asarray([a.date() == unique_date for a in time_datetime])
        time_unique_date = time_datetime[idx_unique_date]
        lon_unique_date = lon[idx_unique_date]
        lat_unique_date = lat[idx_unique_date]
        YMD = time_unique_date[0].date()
        unique_seconds = np.unique(np.asarray([a.hour*3600+a.minute*60+a.second for a in time_unique_date]))
        seconds = np.arange(np.min(unique_seconds),np.max(unique_seconds)+2)
        seconds_since_midnight = [a.hour*3600 + a.minute*60 + a.second + a.microsecond/1000000 for a in time_unique_date]
        idx_time = np.asarray([np.argmin(abs(t - seconds)) for t in seconds_since_midnight])
        tide_time = pyTMD.time.convert_calendar_dates(YMD.year,YMD.month,YMD.day,second=seconds)
        amp,ph = extract_FES_constants(np.atleast_1d(lon_unique_date),
                np.atleast_1d(lat_unique_date), model.model_file, TYPE=model.type,
                VERSION=model.version, METHOD='spline', EXTRAPOLATE=True,
                CUTOFF=5.0,SCALE=model.scale, GZIP=model.compressed)
        DELTAT = calc_delta_time(delta_file, tide_time)
        cph = -1j*ph*np.pi/180.0
        hc = amp*np.exp(cph)
        tmp_tide_heights = np.empty(len(lon_unique_date))
        for i in range(len(lon_unique_date)):
            if np.any(amp[i].mask) == True:
                tmp_tide_heights[i] = np.nan
            else:
                TIDE =         predict_tidal_ts(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,DELTAT=DELTAT[idx_time[i]],CORRECTIONS=model.format)
                MINOR = infer_minor_corrections(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,DELTAT=DELTAT[idx_time[i]],CORRECTIONS=model.format)
                TIDE.data[:] += MINOR.data[:]
                tmp_tide_heights[i] = TIDE.data
        tide_heights[idx_unique_date] = tmp_tide_heights
    return tide_heights