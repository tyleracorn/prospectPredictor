.. _welcome:
.. public

Welcome 
=======

Welcome to prospectPredictor, a python 3.8 module for creating prospectivity predictions. This package is broken up into 3 tools. 1) Prepping GIS shapefiles 2) creating prediction raster templates 3) predicting prospectivity based on shapes of interest

Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

General Package Overview
++++++++++++++++++++++++
the prospectPredictor package is designed to chain together the classes (prepShapes, rasterTemplate, and predictor). Each tool uses functions and wrappers that rely on other python GIS packages, namely geopandas, rasterio, and shapely.

Terms of Use
++++++++++++
prospectPredictor is licensed under the MIT license, which may be found in the included License

.. include:: ../LICENSE
