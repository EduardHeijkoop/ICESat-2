import duckdb
import configparser
import glob
import os
import json

from icesat2.utils.config_utils import get_icesat2_config

class DuckDBMask:
    '''
    Class to use duckdb to mask an input parquet file with a parquet or shapefile, like coasts or building polygons
    '''
    def __init__(self,configuration: configparser.ConfigParser):
        self.configuration_data = configuration
        self.config = get_icesat2_config(configuration)
        self.vector_exts = ('.shp', '.gpkg', '.geojson', '.json', '.fgb', '.gml', '.kml', '.sqlite', '.zip')
        self.geom_col_candidates = ('geometry', 'geom', 'wkb_geometry', 'the_geom', 'wkb', 'shape')

    def mask(self,input_parquet_file,mask_file,output_file=None,keep='inside',verbose=True):
        if output_file is None:
            output_file = input_parquet_file.replace('.parquet','_masked.parquet') if input_parquet_file.endswith('parquet') else f'{input_parquet_file}_masked.parquet'

        con = duckdb.connect()
        con.execute('INSTALL spatial; LOAD spatial;')
        con.execute('SET geometry_always_xy = true')

        is_dir = os.path.isdir(input_parquet_file)
        part_cols = self._get_hive_partitioning(input_parquet_file) if is_dir else []
        src_glob = os.path.join(*[input_parquet_file, '**', '*.parquet']) if is_dir else input_parquet_file
        parquet_src = f'read_parquet({self._sql_str(src_glob)}, hive_partitioning = {str(bool(part_cols)).lower()})'
        x_min, y_min, x_max, y_max = self._get_xy_minmax(con,parquet_src)

        con.execute(f'SET threads = {self.config.N_threads};')  # I/O bound: more threads than cores helps
        con.execute(f'CREATE OR REPLACE TABLE land AS {self._mask_cte_sql(con,mask_file, x_min, y_min, x_max, y_max)}')
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

    def _get_hive_partitioning(self,input_parquet_file):
        files = sorted(glob.glob(os.path.join(input_parquet_file, '**', '*.parquet'), recursive=True))
        if not files:
            raise FileNotFoundError(f'No parquet files under {input_parquet_file}')
        rel = os.path.relpath(os.path.dirname(files[0]), input_parquet_file)
        return [part.split('=', 1)[0] for part in rel.split(os.sep) if '=' in part]

    def _sql_str(self,path):
        '''Quote a path as a SQL string literal.'''
        sql_path = "'" + str(path).replace("'", "''") + "'"
        return sql_path

    def _masked_select(self,src,keep,extra_cols=''):
        '''Point-in-polygon mask written as a join, so DuckDB can use its R-tree spatial join
        instead of testing every point against every polygon.'''
        pt = 'ST_Point(p.lon, p.lat)'
        if keep == 'outside':  # anti-join: points that hit no land polygon
            join, where = 'LEFT JOIN', '\nWHERE l.geom IS NULL'
        elif keep == 'inside':
            join, where = 'JOIN', ''
        else:
            raise ValueError("KEEP must be 'inside' or 'outside'")
        return f'''SELECT p.*{extra_cols}
        FROM {src} p
        {join} land l ON ST_Intersects(l.geom, {pt}){where}'''

    def _mask_cte_sql(self, con, mask_src, xmin, ymin, xmax, ymax, geom_col=None, dissolve=True):
        '''
        CTE = Common Table Expression

        SQL (no trailing semicolon) giving mask polygons clipped to the bbox.

        mask_src can be:
          - a GDAL vector file (.shp, .gpkg, .geojson, .fgb, ...), read with ST_Read
          - a parquet file, a directory of (hive-partitioned) parquet files, a glob or an s3:// path:
              * GeoParquet / native parquet GEOMETRY column -> used directly
              * BLOB column with WKB (e.g. Overture)        -> ST_GeomFromWKB
              * VARCHAR column with WKT                      -> ST_GeomFromText
            If the parquet file has an Overture-style bbox struct (xmin/xmax/ymin/ymax), it is filtered
            first so DuckDB can skip row groups outside the bbox.

        geom_col: geometry column name; auto-detected if None.
        dissolve: union the clipped polygons and split them into single, non-overlapping polygons, so a
                  point matches at most one polygon in the join. Turn off for large building datasets,
                  where the union is slow; then a point on an edge shared by two polygons can appear
                  twice in the 'land' (inner join) output.
        Assumes the mask is in lon/lat (EPSG:4326 / OGC:CRS84).
        '''
        env = f'ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})'
        ext = os.path.splitext(mask_src.rstrip('/'))[1].lower()

        if ext in self.vector_exts:
            from_sql = f'ST_Read({self._sql_str(mask_src)})'
            geom = self._quote_ident(geom_col) if geom_col else 'geom'
            bbox_filter = ''
        else:
            path = os.path.join(mask_src, '**', '*.parquet') if os.path.isdir(mask_src) else mask_src
            from_sql = f'read_parquet({self._sql_str(path)})'
            cols = dict(con.execute(f'SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {from_sql})').fetchall())
            geom = self._parquet_geom_expr(cols, geom_col)
            bbox_filter = self._bbox_struct_filter(cols, xmin, ymin, xmax, ymax)

        # ::GEOMETRY drops a CRS tag (e.g. GEOMETRY('OGC:CRS84') from GeoParquet) so it matches the envelope
        clipped = f'''
            SELECT ST_CollectionExtract(ST_Intersection(g, {env}), 3) AS geom
            FROM (SELECT ({geom})::GEOMETRY AS g FROM {from_sql} {bbox_filter})
            WHERE g && {env}   -- cheap bbox test before the exact clip
        '''
        if dissolve:
            polygons = f'SELECT UNNEST(ST_Dump(ST_Union_Agg(geom)), recursive := true) FROM ({clipped})'
        else:
            polygons = f'SELECT UNNEST(ST_Dump(geom), recursive := true) FROM ({clipped})'
        return f'''
        SELECT geom
        FROM ({polygons})
        WHERE ST_GeometryType(geom) = 'POLYGON'
        '''

    def _parquet_geom_expr(self, cols, geom_col=None):
        '''SQL expression turning the parquet geometry column into GEOMETRY.'''
        names = [geom_col] if geom_col else list(self.geom_col_candidates)
        if geom_col and geom_col not in cols:
            raise ValueError(f'Column {geom_col!r} not found; columns: {list(cols)}')
        # 1) a column already typed GEOMETRY (GeoParquet or native parquet geometry)
        typed = [n for n in names if n in cols and cols[n].upper().startswith('GEOMETRY')]
        if not geom_col:
            typed += [n for n, t in cols.items() if t.upper().startswith('GEOMETRY') and n not in typed]
        if typed:
            return self._quote_ident(typed[0])
        # 2) WKB blob or WKT text
        for n in names:
            if n in cols:
                t = cols[n].upper()
                if t == 'BLOB':
                    return f'ST_GeomFromWKB({self._quote_ident(n)})'
                if t == 'VARCHAR':
                    return f'ST_GeomFromText({self._quote_ident(n)})'
        raise ValueError(f'No geometry column found (tried {names}); columns: {cols}')

    def _bbox_struct_filter(self, cols, xmin, ymin, xmax, ymax):
        '''WHERE clause on an Overture/GeoParquet-1.1 style bbox struct, or '' if there is none.'''
        t = cols.get('bbox', '').upper()
        if not all(k in t for k in ('XMIN', 'XMAX', 'YMIN', 'YMAX')):
            return ''
        return (f'WHERE bbox.xmax >= {xmin} AND bbox.xmin <= {xmax} '
                f'AND bbox.ymax >= {ymin} AND bbox.ymin <= {ymax}')

    def _quote_ident(self, name):
        '''Quote a column name as a SQL identifier.'''
        return '"' + str(name).replace('"', '""') + '"'