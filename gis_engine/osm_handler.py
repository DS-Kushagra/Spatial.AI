# gis_engine/osm_handler.py

import osmnx as ox
import geopandas as gpd

def download_osm_roads(place_name, out_file="data/osm_roads.geojson"):
    G = ox.graph_from_place(place_name, network_type='drive')
    gdf = ox.graph_to_gdfs(G, nodes=False)
    gdf.to_file(out_file, driver="GeoJSON")
    print(f"[OSM] Saved roads to {out_file}")

def download_osm_pois(place_name, tag="school", out_file="data/osm_schools.geojson"):
    pois = ox.geometries_from_place(place_name, tags={tag: True}) # type: ignore
    pois = pois[pois.geometry.notnull()]
    pois.to_file(out_file, driver="GeoJSON")
    print(f"[OSM] Saved {tag} POIs to {out_file}")
