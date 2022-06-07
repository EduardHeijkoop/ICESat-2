Repository to download ICESat-2 ATL03 and ATL12 data.  
  
Set paths & constants in icesat2_config.ini  
  
GCP_Read_Mask_Write_ICESat2_ATL03.py  
  
To run: python GCP_Read_Mask_Write_ICESat2_ATL03.py  
  
This script will download ATL03 geolocated photons over land and apply the following processing:  
    Select high confidence over land  
    Remove photons over water with the point in polygon algorithm  
    Apply a height threshold filter w.r.t. SRTM data  
  
The output is a csv in the form of longitude, latitude, height [above WGS84 ellipsoid], UTC time.  
  
Ocean_Read_Mask_Write_ICESat2_ATL03.py  
  
To run: python Ocean_Read_Mask_Write_ICESat2_ATL03.py  
  
This script will download ATL03 geolocated photons over water and apply the following processing:  
    Select medium & high confidence over water  
    Remove photons over land with the point in polygon algorithm  
    If desired apply geophysical corrections, either GOT4.8 or FES2014 ocean tide models  
    Apply a height threshold filter w.r.t. DTU21 data  
  
The output is a csv in the form of longitude, latitude, height [above WGS84 ellipsoid], UTC time.  

Dependencies:  
GCP:  
Python packages:  
numpy, pandas, geopandas, osgeo/gdal, shapely, h5py  
xml, getpass, socket, subprocess, datetime, configparser  
Datasets:  
EGM96 (will upload to repository)  

Ocean:  
Github:  
Tyler Sutterly's pyTMD:  
https://github.com/tsutterley/pyTMD.git  
Python packages:  
numpy, pandas, geopandas, osgeo/gdal, shapely, h5py  
xml, getpass, socket, subprocess, datetime, configparser  

Datasets:  
DTU21 Mean Sea Surface:  
https://ftp.space.dtu.dk/pub/DTU21/1_MIN/DTU21MSS_1min.mss.nc (will add code to transform to GeoTiff and referencing WGS84 instead of T/P ellipsoid)  