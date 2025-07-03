# test_pipeline.py

import geopandas as gpd
from crs_utils import harmonize_crs

gdf = gpd.read_file(r"D:\Spatial.AI\data\gadm41_IND_shp\gadm41_IND_1.shp")  # ✅ Use your actual file name
gdf = harmonize_crs(gdf)

print("✅ Vector loaded")
print("CRS:", gdf.crs)
print("First rows:")
print(gdf.head())
