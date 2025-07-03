# gis_engine/data_loader.py

import os
import geopandas as gpd
import rasterio

def list_available_layers(data_dir="data"):
    """
    Lists all vector and raster layers available for Spatial.AI
    """
    return [f for f in os.listdir(data_dir) if f.endswith((".geojson", ".shp", ".tif"))]

def load_layer(filename, data_dir="data"):
    """
    Loads a vector or raster layer from Bhoonidhi or other sources
    """
    path = os.path.join(data_dir, filename)
    if filename.endswith((".geojson", ".shp")):
        return gpd.read_file(path)
    elif filename.endswith(".tif"):
        return rasterio.open(path)
    else:
        raise ValueError("Unsupported file format")
