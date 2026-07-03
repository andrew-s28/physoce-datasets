# Expertly Crafted Oceanographic Datasets

PhysOce Datasets is a Python package and command line interface that provides an opionated downloading interface with access to a variety oceanographic datasets. PhysOce Datasets then calculates many derived parameters, such as wind stress and mixed layer depth, according to modern best practices.

## Why PhysOce Datasets?

This package provides an *opinionated* interface which aims to simplify and align access to datasets across providing institutions (NASA, ECMWF, etc.); as such, it is aimed primarily at those looking for streamlined data access. If you are an expert user who wants a lot of control over the details of the download and analysis process, this may not be for you. However, if you want:

- a unified command line and Python interface across datasets
- expert-informed derived variables such as wind stress and eddy kinetic energy
- long time series at a single or a few locations
- the simplest downloading and opening process possible

then this package is designed for you!

## Credentials

Upon usage, this package may prompt for credentials to various data stores. Please refer to their documentation for how to setup and access credentials and how credentials are stored. Once stored, these should not require further maintenance.

- [Copernicus Marine Services](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/how-to-api)
- [NASA Earthdata](https://urs.earthdata.nasa.gov/users/new)
