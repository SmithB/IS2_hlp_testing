# Introduction

This repository contains code aimed at testing existing and prospective algorithms for ICESat-2 data products.  Code and notebooks 
create synthetic low-level products (i.e. the [ATL06 along-track height product](https://nsidc.org/data/atl06/versions/6)) and process
them into higher-level products (i.e. [the ATL11 along-track height-change prodct](https://nsidc.org/data/atl11/versions/6).)  

# Installation

The best way to install the packages is to use conda to install some basic packages, then install the lower-level product ICESat-2 packages
from git.

1. Install gdal and basic python packages:

   conda install gdal numpy scipy suitesparse pip

3. Install the pointCollection package (basic data structures used by this and other higher-level product packages

   git clone https://github.com/SmithB/pointCollection.git ; cd pointCollection; pip install -e . ; cd ..

5. Install the ATL11 package (intermediate-level ICESat-2 products):

   git clone https://github.com/SmithB/ATL11.git ; cd ATL11; pip install -e .; cd ..

7. Install the LSsurf package (height-change gridding):

   git clone https://github.com/SmithB/LSsurf.git ; cd LSsurf; pip install -e .; cd ..

# Data

The main demonstration notebook uses DEM data from Greenland.  There's a script in the main repo directory (get_data.sh) that will 
make a data directory and downlad ArcticDEM data into it, but you may not want to have 1.3 GB of data in your code directory (right?).
To keep data and code separate, make a 'data' somewhere else on your filesystem, and make a symbolic link to it in the top-level repo
directory before running get_data.sh

# Demos

If these instructions have worked, you should be able to run the "Demos.ipynb" script in the notebooks directory.

