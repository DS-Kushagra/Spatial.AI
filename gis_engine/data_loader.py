# gis_engine/data_loader.py

import os
import json
import requests
import geopandas as gpd
import rasterio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from config import BHOONIDHI_API_URL, OSM_API_URL

logger = logging.getLogger(__name__)

class DataLoader:
    """Enhanced data loader with Bhoonidhi and OSM integration"""
    
    def __init__(self, data_dir="data", cache_dir="data/cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Bhoonidhi layer mappings
        self.bhoonidhi_layers = {
            'administrative_boundaries': 'admin_boundaries',
            'land_use': 'landuse',
            'soil': 'soil_data', 
            'elevation': 'dem',
            'water_bodies': 'water',
            'forest_cover': 'forest',
            'urban_areas': 'urban',
            'roads': 'transport',
            'railways': 'railways'
        }

    def list_available_layers(self, include_remote=True):
        """Lists all vector and raster layers available for Spatial.AI"""
        local_files = []
        if self.data_dir.exists():
            local_files = [f.name for f in self.data_dir.glob("*") 
                          if f.suffix in [".geojson", ".shp", ".tif", ".gpkg"]]
        
        available_layers = {
            'local': local_files,
            'bhoonidhi': list(self.bhoonidhi_layers.keys()) if include_remote else [],
            'osm': ['roads', 'buildings', 'pois', 'landuse'] if include_remote else []
        }
        
        return available_layers

    def load_layer(self, filename: str, source: str = "local") -> Union[gpd.GeoDataFrame, rasterio.DatasetReader]:
        """Loads a vector or raster layer from various sources"""
        
        if source == "local":
            return self._load_local_layer(filename)
        elif source == "bhoonidhi":
            return self._load_bhoonidhi_layer(filename)
        elif source == "osm":
            return self._load_osm_layer(filename)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _load_local_layer(self, filename: str):
        """Load layer from local storage"""
        path = self.data_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        if path.suffix in [".geojson", ".shp", ".gpkg"]:
            return gpd.read_file(path)
        elif path.suffix == ".tif":
            return rasterio.open(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_bhoonidhi_layer(self, layer_name: str, bounds: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """Load layer from Bhoonidhi API"""
        
        # Check cache first
        cache_file = self.cache_dir / f"bhoonidhi_{layer_name}.geojson"
        if cache_file.exists():
            logger.info(f"Loading {layer_name} from cache")
            return gpd.read_file(cache_file)
        
        # Mock Bhoonidhi data for demo (real API integration would go here)
        logger.info(f"Generating mock Bhoonidhi data for {layer_name}")
        mock_data = self._generate_mock_bhoonidhi_data(layer_name, bounds)
        
        # Cache the data
        mock_data.to_file(cache_file, driver="GeoJSON")
        
        return mock_data

    def _load_osm_layer(self, layer_type: str, place_name: str = "India", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """Load layer from OpenStreetMap"""
        
        cache_file = self.cache_dir / f"osm_{layer_type}_{place_name.replace(' ', '_')}.geojson"
        if cache_file.exists():
            logger.info(f"Loading OSM {layer_type} from cache")
            return gpd.read_file(cache_file)
        
        # Mock OSM data for demo (real Overpass API integration would go here)
        logger.info(f"Generating mock OSM data for {layer_type}")
        mock_data = self._generate_mock_osm_data(layer_type, bbox)
        
        # Cache the data
        mock_data.to_file(cache_file, driver="GeoJSON")
        
        return mock_data

    def _generate_mock_bhoonidhi_data(self, layer_name: str, bounds: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """Generate mock Bhoonidhi data for demo purposes"""
        from shapely.geometry import Polygon, Point
        import random
        
        if bounds is None:
            # Default to India bounds
            bounds = [68.0, 8.0, 97.0, 37.0]  # [min_lon, min_lat, max_lon, max_lat]
        
        min_lon, min_lat, max_lon, max_lat = bounds
        features = []
        
        if layer_name == 'administrative_boundaries':
            # Generate district boundaries
            for i in range(20):
                center_lon = random.uniform(min_lon, max_lon)
                center_lat = random.uniform(min_lat, max_lat)
                size = random.uniform(0.5, 2.0)
                
                coords = [
                    (center_lon - size, center_lat - size),
                    (center_lon + size, center_lat - size),
                    (center_lon + size, center_lat + size),
                    (center_lon - size, center_lat + size),
                    (center_lon - size, center_lat - size)
                ]
                
                features.append({
                    'geometry': Polygon(coords),
                    'district_name': f"District_{i+1}",
                    'state': random.choice(['Kerala', 'Gujarat', 'Karnataka', 'Maharashtra']),
                    'population': random.randint(100000, 5000000)
                })
                
        elif layer_name == 'land_use':
            # Generate land use polygons
            land_types = ['agricultural', 'forest', 'urban', 'water', 'barren']
            for i in range(100):
                center_lon = random.uniform(min_lon, max_lon)
                center_lat = random.uniform(min_lat, max_lat)
                size = random.uniform(0.1, 0.5)
                
                coords = [
                    (center_lon - size, center_lat - size),
                    (center_lon + size, center_lat - size),
                    (center_lon + size, center_lat + size),
                    (center_lon - size, center_lat + size),
                    (center_lon - size, center_lat - size)
                ]
                
                features.append({
                    'geometry': Polygon(coords),
                    'land_use': random.choice(land_types),
                    'area_ha': random.uniform(10, 1000)
                })
                
        elif layer_name == 'soil':
            # Generate soil data
            soil_types = ['alluvial', 'black', 'red', 'laterite', 'sandy']
            for i in range(50):
                center_lon = random.uniform(min_lon, max_lon)
                center_lat = random.uniform(min_lat, max_lat)
                size = random.uniform(0.2, 0.8)
                
                coords = [
                    (center_lon - size, center_lat - size),
                    (center_lon + size, center_lat - size),
                    (center_lon + size, center_lat + size),
                    (center_lon - size, center_lat + size),
                    (center_lon - size, center_lat - size)
                ]
                
                features.append({
                    'geometry': Polygon(coords),
                    'soil_type': random.choice(soil_types),
                    'ph_level': random.uniform(5.5, 8.5),
                    'fertility': random.choice(['low', 'medium', 'high'])
                })
        
        else:
            # Generic polygon data
            for i in range(30):
                center_lon = random.uniform(min_lon, max_lon)
                center_lat = random.uniform(min_lat, max_lat)
                size = random.uniform(0.1, 0.3)
                
                coords = [
                    (center_lon - size, center_lat - size),
                    (center_lon + size, center_lat - size),
                    (center_lon + size, center_lat + size),
                    (center_lon - size, center_lat + size),
                    (center_lon - size, center_lat - size)
                ]
                
                features.append({
                    'geometry': Polygon(coords),
                    'name': f"{layer_name}_{i+1}",
                    'value': random.uniform(0, 100)
                })
        
        return gpd.GeoDataFrame(features, crs="EPSG:4326")

    def _generate_mock_osm_data(self, layer_type: str, bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """Generate mock OSM data for demo purposes"""
        from shapely.geometry import Point, LineString
        import random
        
        if bbox is None:
            bbox = [68.0, 8.0, 97.0, 37.0]  # India bounds
        
        min_lon, min_lat, max_lon, max_lat = bbox
        features = []
        
        if layer_type == 'roads':
            # Generate road networks
            for i in range(50):
                start_lon = random.uniform(min_lon, max_lon)
                start_lat = random.uniform(min_lat, max_lat)
                end_lon = start_lon + random.uniform(-0.5, 0.5)
                end_lat = start_lat + random.uniform(-0.5, 0.5)
                
                features.append({
                    'geometry': LineString([(start_lon, start_lat), (end_lon, end_lat)]),
                    'highway': random.choice(['primary', 'secondary', 'tertiary', 'residential']),
                    'name': f"Road_{i+1}"
                })
                
        elif layer_type == 'pois':
            # Generate points of interest
            poi_types = ['school', 'hospital', 'bank', 'restaurant', 'fuel']
            for i in range(100):
                lon = random.uniform(min_lon, max_lon)
                lat = random.uniform(min_lat, max_lat)
                
                features.append({
                    'geometry': Point(lon, lat),
                    'amenity': random.choice(poi_types),
                    'name': f"POI_{i+1}"
                })
        
        return gpd.GeoDataFrame(features, crs="EPSG:4326")

# Legacy functions for backward compatibility
def list_available_layers(data_dir="data"):
    """Legacy function - use DataLoader class instead"""
    loader = DataLoader(data_dir)
    return loader.list_available_layers()['local']

def load_layer(filename, data_dir="data"):
    """Legacy function - use DataLoader class instead"""
    loader = DataLoader(data_dir)
    return loader.load_layer(filename)
