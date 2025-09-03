# ICESat-2 Download & Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12691578.svg)](https://doi.org/10.5281/zenodo.12691578)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/license/mit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy)](https://python.org)

## Funding & Support

This work was funded by the NASA Sea Level Change Team (N-SLCT) and performed at the Colorado Center for Astrodynamics Research (CCAR), part of the Ann & H.J. Smead Department of Aerospace Engineering at the University of Colorado Boulder.

<p float "middle">
    <a href="https://www.colorado.edu/aerospace">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/CU_Smead_Logo.png" height=200 alt="CU Aerospace Engineering"/>
    </a>
    <a href="https://www.colorado.edu/ccar">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/CCAR_Logo.jpg" height=200 alt="Colorado Center for Astrodynamics Research"/>
    </a>
</p>


<p float "middle">
    <a href="https://sealevel.nasa.gov">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/NSLCT_Logo.png" height=200 alt="NASA Sea Level Change Team" />
    </a>
</p>

## Installation

Clone the repository:

    git clone https://github.com/EduardHeijkoop/ICESat-2.git

Create a new conda environment with the right packages:

    conda env create -f icesat2.yml

Make sure you have an active [NASA EarthData](https://urs.earthdata.nasa.gov/) account. Other Unix CLI tools that are needed are: `wget` and `unzip`.

### Optional

It is possible to filter the ICESat-2 ATL03 geolocated photons with an _a priori_ height: either the Copernicus DEM over land or a mean sea surface (MSS) product (e.g. DTU21) over water.

- The Copernicus 1° x 1° tiles (30 m resolution) are available on a [public S3 bucket](https://aws.amazon.com/marketplace/pp/prodview-yl3hc4awgb5pu), and to access them the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) must be installed. These heights are referenced to the EGM2008 geoid, as opposed to the WGS 84 ellipsoid for ICESat-2 ATL03. Download the EGM2008 geoid [here](https://s3-eu-west-1.amazonaws.com/download.agisoft.com/gtg/us_nga_egm2008_1.tif) to convert Copernicus heights to reference WGS 84.
- One MSS is DTU21 and it is available [here](https://ftp.space.dtu.dk/pub/DTU21/1_MIN/DTU21MSS_1minWGS.mss.gz). Use `gunzip` to decompress the file and `Convert_MSS_to_geotiff.py` to convert it into a GeoTiff that can be used to sample points. Other MSS products include the CNES CLS and Hybrid models.

## Configuration

### Config File
The scripts in this repository depend on a number of other files & variables that need to be defined in the configuration file. Edit the `icesat2_config.ini` file in this repository or create your own and point to that when you run the script. The structure (e.g. `GCP_PATHS`) must be the same as the original.

General Paths & Constants
- `earthdata_username` : Enter your NASA EarthData username here.
- `osm_shp_file` : OpenStreetMap's land polygons are used to define the boundary between land and water. Click [here](https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip) to download the zip file. Extract this and point to the `land_polygons.shp` file.
- `landmask_c_file` : Point to the `pnpoly_function.c` file that is used to run the landmask algorithm. This repository comes with a file in the `C_Code` directory, so easiest is to point to that. A corresponding `.so` file will be made, ***do not*** point to that.

GCP
- `input_file` : Path to the input file for GCP download. See next section for formatting.
- `icesat2_dir` : Path to directory in which ICESat-2 data will be produced; each location will have its own subdirectory.
- `error_log_file` : Log file that will be appended to when something goes wrong.
- `EGM2008_path` : Path to the EGM2008 file when the optional `--copernicus` flag is used.
- `landmask_inside_flag` : Value for landmask to be used: 1 means only photons over land (default) and 0 means no photons over land.
- `copernicus_threshold` : Value for vertical threshold between ICESat-2 and Copernicus DEM. Heights where `abs(h_icesat2 - h_copernicus) > copernicus_threshold` will be discarded. The MERIT DEM can optionally be downloaded with ICESat-2 data, but the Copernicus product is a better representation of the Earth's surface at a higher spatial resolution.

Ocean
- `input_file` : Path to the input file for Ocean download. See next section for formatting.
- `icesat2_dir` : Path to directory in which ICESat-2 data will be produced; each location will have its own subdirectory.
- `error_log_file` : Log file that will be appended to when something goes wrong.
- `mss_path` : Path to the MSS file when the optional `--mss` flag is used.
- `model_dir` : Path to directory that contains FES2014/22 model files.
- `landmask_inside_flag` : 


### Input File

Both the GCP and Ocean ATL03 scripts require an input file that has the location's name, defines the bounding box, and optionally has a temporal subset. The first line of the input file is the header and the variables must be in the following order: name, lon_min, lon_max, lat_min, lat_max, t_start, t_end.

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/ICESat2_Input.png" alt="ICESat-2 Input File" />
    </a>
</p>


## Usage

### ICESat-2 for Ground Control Points (GCPs)
To run:

    python GCP_Read_Mask_Write_ICESat2_ATL03.py

The default output will be a csv file with longitude, latitude and (WGS 84) height.

Optional flags:

    --config <configuration_file> : Path to configuration file. Default will be icesat2_config.ini in the same directory.
    --landmask : Toggle to creaete a mask for ATL03 photons over land/water and subsequently select those over land only.
    --time : Toggle to add a column of timestamps for each ATL03 photon. These will be UTC and in the format "YYYY-MM-DD hh:mm:ss.ssssss".
    --beams : Toggle to add a column of beam names for each ATL03 photon. Format will be "gtNX", where N is an integer (1,2,3) and X is the side (l/r), e.g. gt2r.
    --strength <strong/weak/all> : Select beam strength, either strong, weak or all. Default is strong.
    --weight : Toggle to incorporate the weight parameter, as defined by the ATL03 ATBD (section 5.2). When selected, only photons with weights greater than 80% will be used.
    --sigma : Toggle to add a column of sigma values for each ATL03 photon. Note: this is required if ATL03 is used to co-register DSMs, as this value is needed to calculate final DSM uncertainty.
    --fpb : Toggle to incorporate the first photon bias correction.
    --N_cpus <value> : Select the number of CPU cores to perform the landmask with. Default is 1.
    --copernicus : Toggle to filter photons based on the a priori Copernicus DEM product. Outliers greater than the value set in the configuration file are removed.
    --keep_files : Toggle to keep the Copernicus DEM file rather than delete it afterwards.

I recommend running the script as follows:

    python GCP_Read_Mask_Write_ICESat2_ATL03.py --landmask --time --beams --sigma --N_cpus <N_cpus>

where `N_cpus` can be determined based on the complexity of the topography and the size of the area of interest.

### Filter ICESat-2 with Sentinel-2

As an optional step, an ICESat-2 ATL03 point cloud over land can be filtered with spatially and temporally coincident Sentinel-2 multispectral imagery.

`gcloud`

    conda install -c conda-forge earthengine google-auth

### ICESat-2 over Water
To run:

    python Ocean_Read_Mask_Write_ICESat2_ATL03.py

The default output will be a csv file with longitude, latitude and (WGS 84) height.

Optional flags:


### Convert Point Cloud into Geometry
ATL03 point clouds can become very dense due to the 10 kHz pulse frequency. When plotting (e.g. in QGIS) these dense point clouds may end up looking like lines. To lighten the computational load, this script converts point clouds into LineString geometries which are much easier to plot. Requires that the ICESat-2 csv has a "time" column to calculate jumps.

    python Convert_ATL03_csv_to_shp.py --input_file <input_file>

Optional flags:

    --dt_threshold <value> : Select value after which a jump is implemented. Default is 0.01 seconds. 
    --output_format <shp/geojson> : Select output format to be either a shapefile or geojson. Default is shapefile.

### Citation

To cite this repository, please use:

Eduard Heijkoop. (2024). EduardHeijkoop/ICESat-2: First release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.12691578


### Acknowledgments

I'd like to acknowledge the following organizations for their open source and/or freely available tools that help make this software work:
- OpenStreetMap and its contributors for producting the coastline.
- Sinergise for hosting Copernicus on AWS S3.
- Danish Technical University (DTU) for creating DTU21 MSS.


### To Do
- [ ] Upload `Convert_MSS_to_geotiff.py` to repository.
- [ ] Rewrite `Ocean_Read_Mask_Write_ICESat2_ATL03.py` to numpy v2 and pyTMD v2.
- [ ] Include other MSS products.
- [ ] Include flag to select either FES2014 or FES2022.
