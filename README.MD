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

Edit the icesat2.yml file to set the prefix to your machine's, run `conda/mamba env list` to see the structure.

Create a new conda environment with the right packages:

    conda create -f icesat2.yml
    mamba create -f icesat2.yml

Activate environment:

    conda activate icesat2
    mamba activate icesat2

Install within environment:

    pip install -e .

Make sure you have an active [NASA EarthData](https://urs.earthdata.nasa.gov/) account.

<!-- ### Optional

It is possible to filter the ICESat-2 ATL03 geolocated photons with an _a priori_ height: either the Copernicus DEM over land or a mean sea surface (MSS) product (e.g. DTU21) over water.

- The Copernicus 1° x 1° tiles (30 m resolution) are available on a [public S3 bucket](https://aws.amazon.com/marketplace/pp/prodview-yl3hc4awgb5pu), and to access them the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) must be installed. These heights are referenced to the EGM2008 geoid, as opposed to the WGS 84 ellipsoid for ICESat-2 ATL03. Download the EGM2008 geoid [here](https://s3-eu-west-1.amazonaws.com/download.agisoft.com/gtg/us_nga_egm2008_1.tif) to convert Copernicus heights to reference WGS 84.
- One MSS is DTU21 and it is available [here](https://ftp.space.dtu.dk/pub/DTU21/1_MIN/DTU21MSS_1minWGS.mss.gz). Use `gunzip` to decompress the file and `Convert_MSS_to_geotiff.py` to convert it into a GeoTiff that can be used to sample points. Other MSS products include the CNES CLS and Hybrid models. -->

## Configuration

### Config File
The scripts in this repository depend on a number of other files & variables that need to be defined in the configuration file. Edit the `icesat2_config.ini` file in this repository (under `src/icesat2/config/icesat2_config.ini`) or create your own and point to that when you run the script. The structure (e.g. `ATL03_OPTIONS`) must be the same as the original.

General Paths & Constants
- `earthdata_username` : Enter your NASA EarthData username here.
- `project_dir` : Directory where output will go. Each new run will create its own subdirectory as defined by the `output_prefix` flag.
- `osm_shp_file` : OpenStreetMap's land polygons are used to define the boundary between land and water. Click [here](https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip) to download the zip file. Extract this and point to the `land_polygons.shp` file.

ICESat-2
- `N_threads` : How many threads to use for masking (and later on sampling).
- `landmask_flag` : Toggle to do masking or not.
- `N_split_years` : How many years to split request into. This allows for splitting of the job on the NSIDC's end.

ATL03
- `beam_strength` : Select strong/weak/all beam strengths.
- `photon_confidence` : Select photon confidence level. Options are high/medium, and medium will include all high confidence photons too.
- `high_quality_flag` : Select high quality photons only. 
- `version` : ATL03 version to use. Currently (Nov 2026) this is 7, with v6 being deprecated.

## Usage

### ICESat-2 for Ground Control Points (GCPs)
To run:

    python Download_ATL03.py --extents lon_min lat_min lon_max lat_max --output_prefix location_name [optional flags]

The output will be a parquet file with UTC time, longitude, latitude and (WGS 84) height, plus sigma, RGT (reference ground track), cycle, geoid height, geoid free2mean correction, equilibrium tide, DAC & ocean tide. The parquet file will be saved in a subdirectory within the output directory specified in the config file. It will be hive-partitioned by date, strength and beam for faster loading and filtering.

Optional flags:

    --topography <topography_file> : Path to a topography file, from which lon/lat extents will be extracted. Use this instead of --extents.
    --surface_type <surface_type> : Select surface type for ATL03 (land or ocean), default is land.
    --config <configuration_file> : Path to configuration file. Default will be icesat2_config.ini in the src/icesat2/config directory.
    --verbose : Toggle to run in verbose mode, i.e. print more info.
    --product <product> : Select ICESat-2 product, so far only ATL03 is implemented, but ATL06 (land ice) and ATL08 (land heights) will be implemented soon.


<!-- ### Filter ICESat-2 with Sentinel-2

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
    --output_format <shp/geojson> : Select output format to be either a shapefile or geojson. Default is shapefile. -->

### Citation

To cite this repository, please use:

Eduard Heijkoop. (2024). EduardHeijkoop/ICESat-2: First release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.12691578 (to be updated to version 2)


### Acknowledgments

I'd like to acknowledge the following organizations for their open source and/or freely available tools that help make this software work:
- OpenStreetMap and its contributors for producting the coastline.


### To Do
<!-- - [ ] Upload `Convert_MSS_to_geotiff.py` to repository.
- [ ] Rewrite `Ocean_Read_Mask_Write_ICESat2_ATL03.py` to numpy v2 and pyTMD v2.
- [ ] Include other MSS products.
- [ ] Include flag to select either FES2014 or FES2022. -->
- [ ] Re-implement filtering with topography and mean sea surface products
- [ ] Update citation
- [ ] Implement ATL06
- [ ] Implement ATL08