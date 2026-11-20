'''
Fast vectorised sampling of a GeoTIFF (any CRS, e.g. a local UTM zone) at lon/lat points.

    import polars as pl
    from sample_raster import sample_raster, sample_raster_parallel

    df = pl.read_parquet('atl03_parquet/*.parquet', columns=['lon', 'lat', 'h_ph', 'geoid'])
    lon, lat = df['lon'].to_numpy(), df['lat'].to_numpy()

    dem = sample_raster(lon, lat, 'glo30_utm48n.tif')                    # single core
    dem = sample_raster_parallel(lon, lat, 'glo30_utm48n.tif', workers=8)  # threads (default)
    dem = sample_raster_parallel(lon, lat, 'glo30_utm48n.tif', workers=8, executor='process')

    dh = (df['h_ph'] - df['geoid']).to_numpy() - dem   # GLO-30 is EGM2008, like the ATL03 geoid

Threads are the recommended default: pyproj, GDAL reads and scipy's map_coordinates
all release the GIL, so threads scale with cores and share memory with no copying.
Processes work too, but every chunk of coordinates and results is pickled between processes.

The parallel version
  1. transforms lon/lat -> pixel coordinates in parallel chunks,
  2. groups points into horizontal bands of raster rows (an O(n) radix sort),
  3. each worker reads only its band's window and interpolates it.
Each raster row is read about once and peak memory stays bounded, whatever order the points come in.
'''
import os
import configparser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import rasterio as rio
import pyproj
# from rasterio.windows import Window
from scipy.ndimage import map_coordinates
from icesat2.utils.config_utils import get_icesat2_config



class RasterSampler:
    def __init__(self,configuration: configparser.ConfigParser):
        self.configuration_data = configuration
        self.config = get_icesat2_config(configuration)

# ---------------------------------------------------------------- serial version
def sample_raster(lon, lat, tif_path, band=1, method='bilinear', src_crs='EPSG:4326'):
    '''Return raster values at lon/lat (NaN outside the raster or on nodata).

    method: 'bilinear' or 'nearest'.
    '''
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    out = np.full(lon.shape, np.nan)

    with rio.open(tif_path) as src:
        col, row = _to_pixel(lon, lat, src_crs, src.crs.to_wkt(), tuple(src.transform)[:6])
        inside = (col >= 0) & (col < src.width) & (row >= 0) & (row < src.height)
        if not inside.any():
            return out
        col, row = col[inside], row[inside]
        r0, r1, c0, c1 = _window_bounds(row, col, src.height, src.width)

    out[inside] = _read_and_interp(tif_path, band, r0, r1, c0, c1, row, col, method)
    return out


# ---------------------------------------------------------------- parallel version
def sample_raster_parallel(lon, lat, tif_path, band=1, method='bilinear', src_crs='EPSG:4326',
                           workers=None, executor='thread', max_band_pixels=50_000_000):
    '''Parallel equivalent of sample_raster(); same output.

    workers: number of threads/processes (default: all cores).
    executor: 'thread' (recommended) or 'process'.
    max_band_pixels: upper bound on the pixels one worker reads at once (memory cap per worker;
        50M float64 pixels = 400 MB).
    '''
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    n = lon.size
    out = np.full(lon.shape, np.nan)
    if n == 0:
        return out
    workers = workers or os.cpu_count()
    pool_cls = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}[executor]

    with rio.open(tif_path) as src:
        height, width = src.height, src.width
        dst_wkt = src.crs.to_wkt()
        affine6 = tuple(src.transform)[:6]

    with pool_cls(workers) as pool:
        # 1. lon/lat -> fractional pixel coordinates, in parallel chunks
        edges = np.linspace(0, n, min(workers * 4, n) + 1).astype(np.int64)
        futs = [pool.submit(_to_pixel, lon[a:b], lat[a:b], src_crs, dst_wkt, affine6)
                for a, b in zip(edges[:-1], edges[1:])]
        col = np.empty(n)
        row = np.empty(n)
        for (a, b), fut in zip(zip(edges[:-1], edges[1:]), futs):
            col[a:b], row[a:b] = fut.result()

        inside = np.flatnonzero((col >= 0) & (col < width) & (row >= 0) & (row < height))
        if inside.size == 0:
            return out
        col, row = col[inside], row[inside]

        # 2. split the covered rows into bands: enough bands to keep all workers busy,
        #    and small enough that a band's window stays under max_band_pixels
        rmin, rmax = int(row.min()), int(row.max()) + 1
        cspan = int(col.max()) - int(col.min()) + 3
        band_h = max(1, min(-(-(rmax - rmin) // (workers * 4)), max_band_pixels // cspan))
        band_id = (row.astype(np.int64) - rmin) // band_h
        n_bands = int(band_id.max()) + 1
        # int16 keys get numpy's O(n) radix sort; int32 falls back to a ~4x slower merge sort
        band_id = band_id.astype(np.int16 if n_bands < 2**15 else np.int32)
        order = np.argsort(band_id, kind='stable')
        bounds = np.searchsorted(band_id[order], np.arange(n_bands + 1))

        # 3. one task per band: read just that window and interpolate
        if executor == 'thread':
            # threads share memory: gather and scatter inside the workers too (indices are disjoint)
            def job(idx):
                r, c = row[idx], col[idx]
                r0, r1, c0, c1 = _window_bounds(r, c, height, width)
                out[inside[idx]] = _read_and_interp(tif_path, band, r0, r1, c0, c1, r, c, method)

            futs = [pool.submit(job, order[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            for fut in futs:
                fut.result()  # re-raises any worker exception
        else:
            tasks = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b == a:
                    continue
                idx = order[a:b]
                r, c = row[idx], col[idx]
                r0, r1, c0, c1 = _window_bounds(r, c, height, width)
                tasks.append((idx, pool.submit(_read_and_interp, tif_path, band, r0, r1, c0, c1, r, c, method)))
            for idx, fut in tasks:
                out[inside[idx]] = fut.result()
    return out


# ---------------------------------------------------------------- helpers (top level so processes can pickle them)
def _to_pixel(lon, lat, src_crs, dst_wkt, affine6):
    '''lon/lat -> fractional (col, row); integer values are pixel corners.'''
    x, y = pyproj.Transformer.from_crs(src_crs, dst_wkt, always_xy=True).transform(lon, lat)
    a, b, c, d, e, f = _invert_affine(affine6)
    return a * x + b * y + c, d * x + e * y + f


def _invert_affine(t):
    a, b, c, d, e, f = t
    det = a * e - b * d
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    return ia, ib, -(ia * c + ib * f), id_, ie, -(id_ * c + ie * f)


def _window_bounds(row, col, height, width):
    '''Pixel window covering the points plus a 1-pixel margin for bilinear neighbours.'''
    r0 = max(int(np.floor(row.min())) - 1, 0)
    c0 = max(int(np.floor(col.min())) - 1, 0)
    r1 = min(int(np.ceil(row.max())) + 2, height)
    c1 = min(int(np.ceil(col.max())) + 2, width)
    return r0, r1, c0, c1


def _read_and_interp(tif_path, band, r0, r1, c0, c1, row, col, method):
    # each call opens its own handle: rasterio datasets must not be shared between threads
    with rio.open(tif_path) as src:
        arr = src.read(band, window=rio.windows.Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    order = 1 if method == 'bilinear' else 0
    # map_coordinates indexes pixel centres, hence the -0.5
    coords = np.vstack([row - r0 - 0.5, col - c0 - 0.5])
    # return map_coordinates(arr, coords, order=order, mode='nearest', prefilter=False)
    return map_coordinates(arr, coords, order=order, mode='constant', prefilter=False,cval=-9999.0)
