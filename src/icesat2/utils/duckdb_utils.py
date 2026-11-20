import duckdb
import configparser
import glob
import os


from icesat2.utils.config_utils import get_icesat2_config

class DuckDBLandmask:
    def __init__(self,configuration: configparser.ConfigParser):
        self.configuration_data = configuration
        self.config = get_icesat2_config(configuration)

    def mask(self,parquet_file,keep='land',verbose=True):
        output_file = parquet_file.replace('.parquet','_masked.parquet') if parquet_file.endswith('parquet') else f'{parquet_file}_masked.parquet' 

        con = duckdb.connect()
        con.execute('INSTALL spatial; LOAD spatial;')
        con.execute('SET geometry_always_xy = true')

        is_dir = os.path.isdir(parquet_file)
        part_cols = self._get_hive_partitioning(parquet_file) if is_dir else []
        src_glob = os.path.join(*[parquet_file, '**', '*.parquet']) if is_dir else parquet_file
        parquet_src = f'read_parquet({self._sql_str(src_glob)}, hive_partitioning = {str(bool(part_cols)).lower()})'
        x_min, y_min, x_max, y_max = self._get_xy_minmax(con,parquet_src)

        con.execute(f'SET threads = {self.config.N_threads};')  # I/O bound: more threads than cores helps
        con.execute(f'CREATE OR REPLACE TABLE land AS {self._land_cte_sql(self.config.osm_shp_file, x_min, y_min, x_max, y_max)}')
        con.execute('CREATE INDEX land_rtree ON land USING RTREE (geom)')
        # print('land polygons in bbox:', con.execute('SELECT count(*) FROM land').fetchone()[0])
        options = ['FORMAT parquet', 'COMPRESSION zstd']
        if part_cols:
            options += [f'PARTITION_BY ({", ".join(part_cols)})', 'OVERWRITE_OR_IGNORE true']
        con.execute(f'COPY (\n{self._masked_select(parquet_src, keep)}\n) TO {self._sql_str(output_file)} ({", ".join(options)})')

        # do before/after analysis
        if verbose == True:
            n_in = con.execute(f'SELECT count(*) FROM {parquet_src}').fetchone()[0]
            out_glob = os.path.join(output_file, '**', '*.parquet') if part_cols else output_file
            n_out = con.execute(
                f'SELECT count(*) FROM read_parquet({self._sql_str(out_glob)}, hive_partitioning = {str(bool(part_cols)).lower()})'
            ).fetchone()[0]
            print(f'{n_in} -> {n_out} points written to {output_file}' + (f' (partitioned by {part_cols})' if part_cols else ''))
        return output_file



    def _get_xy_minmax(self,con,parquet_src):
        x_min, y_min, x_max, y_max = con.execute(
            f'SELECT min(lon), min(lat), max(lon), max(lat) FROM {parquet_src}'
        ).fetchone()
        x_min = round(x_min - self.config.landmask_shp_buffer, 6)
        y_min = round(y_min - self.config.landmask_shp_buffer, 6)
        x_max = round(x_max + self.config.landmask_shp_buffer, 6)
        y_max = round(y_max + self.config.landmask_shp_buffer, 6)
        return x_min, y_min, x_max, y_max

        # con.execute(f'''
        #     COPY (
        #         SELECT id,class,geometry
        #         FROM read_parquet({file_list!r})
        #         WHERE bbox.xmax >= {bbox.bounds[0]} AND bbox.xmin <= {bbox.bounds[2]}
        #         AND bbox.ymax >= {bbox.bounds[1]} AND bbox.ymin <= {bbox.bounds[3]}
        #     ) TO '{output_file}' (FORMAT parquet);
        # ''')

    def _get_hive_partitioning(self,parquet_file):
        files = sorted(glob.glob(os.path.join(parquet_file, '**', '*.parquet'), recursive=True))
        if not files:
            raise FileNotFoundError(f'No parquet files under {parquet_file}')
        rel = os.path.relpath(os.path.dirname(files[0]), parquet_file)
        return [part.split('=', 1)[0] for part in rel.split(os.sep) if '=' in part]

    def _sql_str(self,path):
        '''Quote a path as a SQL string literal.'''
        sql_path = "'" + str(path).replace("'", "''") + "'"
        return sql_path

    def _land_cte_sql(self,shp, xmin, ymin, xmax, ymax):
        '''
        SQL (no trailing semicolon) giving land polygons clipped to the bbox, dissolved and split into
        single, non-overlapping polygons (so a point matches at most one polygon in the join).
        spatial_filter = ST_AsWKB(ST_MakeEnvelope(min_x, min_y, max_x, max_y))
        '''
        env = f'ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})'
        return f'''
        SELECT geom
        FROM (
            SELECT UNNEST(ST_Dump(ST_Union_Agg(geom)), recursive := true)
            FROM (
                SELECT ST_CollectionExtract(ST_Intersection(geom, {env}), 3) AS geom
                FROM ST_Read({self._sql_str(shp)})
                WHERE geom && {env}   -- cheap bbox test before the exact clip
            )
        )
        WHERE ST_GeometryType(geom) = 'POLYGON'
        '''
    def _masked_select(self,src,keep,extra_cols=''):
        '''Point-in-polygon mask written as a join, so DuckDB can use its R-tree spatial join
        instead of testing every point against every polygon.'''
        pt = 'ST_Point(p.lon, p.lat)'
        if keep == 'water':  # anti-join: points that hit no land polygon
            join, where = 'LEFT JOIN', '\nWHERE l.geom IS NULL'
        elif keep == 'land':
            join, where = 'JOIN', ''
        else:
            raise ValueError("KEEP must be 'water' or 'land'")
        return f'''SELECT p.*{extra_cols}
        FROM {src} p
        {join} land l ON ST_Intersects(l.geom, {pt}){where}'''
