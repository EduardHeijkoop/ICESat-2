import numpy as np
import shapely
import pandas as pd
import geopandas as gpd
import argparse
import warnings

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###
###Last updated: 2023-05-25###
#This script converts an ICESat-2 ATL03 csv file to a shapefile or geojson file.
#Uses shapely LineStrings to approximate points that are closer than a given time threshold, reducing file size significantly.

def df_to_gdf(df,dt_threshold=0.01):
    '''
    dt_threshold given in seconds, then converted to ns
    Requires dataframe to have lon, lat, beam and time columns
    '''
    dt_threshold = dt_threshold / 1e-9
    gdf = gpd.GeoDataFrame()
    df['date'] = [t[:10] for t in df.time]
    df['t_datetime'] = pd.to_datetime(df.time)
    unique_dates = np.unique(df.date)
    for ud in unique_dates:
        df_date = df[df.date == ud].copy()
        unique_beams = np.unique(df_date.beam)
        for beam in unique_beams:
            df_beam = df_date[df_date.beam == beam].copy()
            t_datetime = np.asarray(df_beam.t_datetime)
            dt = np.asarray(t_datetime[1:] - t_datetime[:-1])
            dt = np.append(0,dt).astype(int)
            idx_jump_orig = np.atleast_1d(np.argwhere(dt > dt_threshold).squeeze())
            idx_jump = np.concatenate([[0],idx_jump_orig,[len(df_beam)-1]])
            for k in range(len(idx_jump)-1):
                geom = shapely.geometry.LineString([(df_beam.lon.iloc[idx_jump[k]],df_beam.lat.iloc[idx_jump[k]]),(df_beam.lon.iloc[idx_jump[k+1]-1],df_beam.lat.iloc[idx_jump[k+1]-1])])
                tmp_gdf = gpd.GeoDataFrame(pd.DataFrame({'date':[ud],'beam':[beam]}),geometry=[geom],crs='EPSG:4326')
                gdf = pd.concat((gdf,tmp_gdf)).reset_index(drop=True)
    return gdf

def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to input file.')
    parser.add_argument('--dt_threshold',help='Time threshold for jump in seconds.',default=0.01,type=float)
    parser.add_argument('--output_format',help='Output format.',default='shp',choices=['shp','geojson'])
    args = parser.parse_args()

    input_file = args.input_file
    dt_threshold = args.dt_threshold
    output_format = args.output_format

    df = pd.read_csv(input_file)
    gdf_atl03 = df_to_gdf(df,dt_threshold=dt_threshold)
    gdf_atl03.to_file(input_file.replace('.txt',f'.{output_format}'))


if __name__ == '__main__':
    main()
