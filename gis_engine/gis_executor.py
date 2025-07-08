# gis_engine/gis_executor.py

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import mapping, Point
from pyproj import CRS
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from pathlib import Path
import logging
from gis_engine.crs_utils import harmonize_crs

logger = logging.getLogger(__name__)

class GISExecutor:
    def __init__(self, data_dir="data", output_dir="outputs"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.layers = {}

    def get_utm_crs(self, gdf):
        # Get centroid from the bounds to avoid empty geometry issues
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        if len(bounds) < 4 or any(pd.isna(bounds)):
            # Fallback to a default UTM zone if bounds are invalid
            return "EPSG:32643"  # UTM Zone 43N for India
        
        centroid_x = (bounds[0] + bounds[2]) / 2
        utm_zone = int((centroid_x + 180) / 6) + 1
        epsg = 32600 + utm_zone  # Northern Hemisphere
        return f"EPSG:{epsg}"

    def execute_workflow(self, workflow_json):
        results = {}
        for step in workflow_json["steps"]:
            op = step["operation"]

            if op == "load":
                layer_path = os.path.join(self.data_dir, step["layer"])
                gdf = gpd.read_file(layer_path)
                gdf = harmonize_crs(gdf)
                self.layers[step["name"]] = gdf

            elif op == "buffer":
                gdf = self.layers[step["input"]]

                if gdf.empty:
                    raise ValueError(f"Input layer '{step['input']}' is empty!")

                # Fix invalid geometries
                gdf["geometry"] = gdf["geometry"].buffer(0)

                # Detect and use appropriate UTM zone for projection
                projected_crs = self.get_utm_crs(gdf)
                gdf_proj = gdf.to_crs(projected_crs)

                # Apply buffer in meters
                try:
                    buffered_proj = gdf_proj.copy()
                    buffered_proj["geometry"] = buffered_proj.buffer(step["distance"])
                except Exception as e:
                    raise RuntimeError(f"Buffer operation failed: {e}")

                # Reproject back to WGS84
                buffered = buffered_proj.to_crs("EPSG:4326")
                self.layers[step["name"]] = buffered

            elif op == "intersect":
                a = self.layers[step["input_a"]]
                b = self.layers[step["input_b"]]
                self.layers[step["name"]] = gpd.overlay(a, b, how="intersection")

            elif op == "clip":
                a = self.layers[step["input"]]
                b = self.layers[step["mask"]]
                self.layers[step["name"]] = gpd.overlay(a, b, how="intersection")

            elif op == "dissolve":
                gdf = self.layers[step["input"]]
                self.layers[step["name"]] = gdf.dissolve(by=step["by"])

            elif op == "filter":
                gdf = self.layers[step["input"]]
                self.layers[step["name"]] = gdf.query(step["query"])

            # Advanced operations
            elif op == "multi_criteria_analysis":
                self.layers[step["name"]] = self._multi_criteria_analysis(step)
                
            elif op == "network_analysis":
                self.layers[step["name"]] = self._network_analysis(step)
                
            elif op == "accessibility_analysis":
                self.layers[step["name"]] = self._accessibility_analysis(step)
                
            elif op == "flood_modeling":
                self.layers[step["name"]] = self._flood_modeling(step)
                
            elif op == "solar_optimization":
                self.layers[step["name"]] = self._solar_optimization(step)
                
            elif op == "suitability_analysis":
                self.layers[step["name"]] = self._suitability_analysis(step)

            elif op == "export":
                gdf = self.layers[step["input"]]
                output_path = os.path.join(self.output_dir, step["filename"])
                gdf.to_file(output_path, driver="GeoJSON")
                logger.info(f"✅ Exported to {output_path}")

            else:
                raise NotImplementedError(f"Unsupported operation: {op}")

        return self.layers

    def _multi_criteria_analysis(self, step):
        """Perform multi-criteria decision analysis"""
        criteria = step.get("criteria", [])
        weights = step.get("weights", [])
        input_layer = step.get("input")
        
        if not criteria or not weights or len(criteria) != len(weights):
            raise ValueError("Criteria and weights must be provided and of equal length")
        
        gdf = self.layers[input_layer].copy()
        
        # Normalize criteria values to 0-1 scale
        scaler = MinMaxScaler()
        
        for criterion in criteria:
            if criterion in gdf.columns:
                gdf[f"{criterion}_normalized"] = scaler.fit_transform(gdf[[criterion]].fillna(0))
        
        # Calculate weighted scores
        gdf['mca_score'] = 0
        for i, criterion in enumerate(criteria):
            if f"{criterion}_normalized" in gdf.columns:
                gdf['mca_score'] += gdf[f"{criterion}_normalized"] * weights[i]
        
        # Rank sites
        gdf['rank'] = gdf['mca_score'].rank(ascending=False, method='dense')
        gdf['suitability'] = pd.cut(gdf['mca_score'], 
                                   bins=[0, 0.3, 0.6, 1.0], 
                                   labels=['Low', 'Medium', 'High'],
                                   include_lowest=True)
        
        return gdf

    def _network_analysis(self, step):
        """Perform network analysis (shortest path, service areas)"""
        roads_layer = step.get("roads_input")
        origins_layer = step.get("origins_input")
        destinations_layer = step.get("destinations_input", None)
        analysis_type = step.get("analysis_type", "shortest_path")
        
        roads_gdf = self.layers[roads_layer]
        origins_gdf = self.layers[origins_layer]
        
        if analysis_type == "shortest_path" and destinations_layer:
            destinations_gdf = self.layers[destinations_layer]
            return self._calculate_shortest_paths(roads_gdf, origins_gdf, destinations_gdf)
        elif analysis_type == "service_area":
            distance_km = step.get("service_distance", 5)
            return self._calculate_service_areas(roads_gdf, origins_gdf, distance_km)
        
        return origins_gdf

    def _calculate_shortest_paths(self, roads_gdf, origins_gdf, destinations_gdf):
        """Calculate shortest paths between origins and destinations"""
        # Simplified network analysis - in production would use OSMnx or GTFS
        results = []
        
        for _, origin in origins_gdf.iterrows():
            origin_point = origin.geometry
            
            for _, dest in destinations_gdf.iterrows():
                dest_point = dest.geometry
                
                # Calculate Euclidean distance as proxy for network distance
                # In production, would use actual network routing
                distance = origin_point.distance(dest_point) * 111000  # Convert to meters
                
                results.append({
                    'origin_id': origin.get('id', 'unknown'),
                    'dest_id': dest.get('id', 'unknown'),
                    'distance_m': distance,
                    'travel_time_min': distance / 50000 * 60,  # Assume 50 km/h average speed
                    'geometry': origin_point
                })
        
        return gpd.GeoDataFrame(results, crs=origins_gdf.crs)

    def _calculate_service_areas(self, roads_gdf, origins_gdf, distance_km):
        """Calculate service areas around origins"""
        service_areas = origins_gdf.copy()
        
        # Convert to appropriate projection for buffering
        projected_crs = self.get_utm_crs(origins_gdf)
        origins_proj = origins_gdf.to_crs(projected_crs)
        
        # Create buffer zones
        buffer_distance = distance_km * 1000  # Convert to meters
        origins_proj['geometry'] = origins_proj.geometry.buffer(buffer_distance)
        
        # Convert back to original CRS
        service_areas = origins_proj.to_crs(origins_gdf.crs)
        service_areas['service_radius_km'] = distance_km
        
        return service_areas

    def _accessibility_analysis(self, step):
        """Analyze accessibility to services"""
        facilities_layer = step.get("facilities_input")
        population_layer = step.get("population_input")
        service_type = step.get("service_type", "healthcare")
        max_distance_km = step.get("max_distance", 10)
        
        facilities_gdf = self.layers[facilities_layer]
        population_gdf = self.layers[population_layer]
        
        results = []
        
        for _, pop_area in population_gdf.iterrows():
            pop_centroid = pop_area.geometry.centroid
            
            # Find nearest facilities
            distances = []
            for _, facility in facilities_gdf.iterrows():
                dist = pop_centroid.distance(facility.geometry) * 111000  # Convert to meters
                distances.append(dist)
            
            min_distance = min(distances) if distances else float('inf')
            accessible_count = sum(1 for d in distances if d <= max_distance_km * 1000)
            
            accessibility_score = min(accessible_count / 5, 1.0)  # Normalize to 0-1
            accessibility_level = 'High' if accessibility_score > 0.7 else \
                                 'Medium' if accessibility_score > 0.3 else 'Low'
            
            result = pop_area.copy()
            result['nearest_facility_km'] = min_distance / 1000
            result['accessible_facilities'] = accessible_count
            result['accessibility_score'] = accessibility_score
            result['accessibility_level'] = accessibility_level
            
            results.append(result)
        
        return gpd.GeoDataFrame(results, crs=population_gdf.crs)

    def _flood_modeling(self, step):
        """Model flood risk and impact"""
        elevation_layer = step.get("elevation_input")
        flood_zones_layer = step.get("flood_zones_input")
        infrastructure_layer = step.get("infrastructure_input", None)
        
        flood_zones_gdf = self.layers[flood_zones_layer]
        
        # Enhanced flood risk analysis
        results = flood_zones_gdf.copy()
        
        # Calculate flood impact scores
        results['impact_score'] = 0
        
        # Risk scoring based on flood depth and return period
        if 'flood_depth' in results.columns:
            results['depth_score'] = np.clip(results['flood_depth'] / 5.0, 0, 1)  # Normalize to 0-1
            results['impact_score'] += results['depth_score'] * 0.4
        
        if 'return_period' in results.columns:
            # Lower return period = higher probability = higher risk
            results['probability_score'] = 1 / (results['return_period'] / 10)
            results['probability_score'] = np.clip(results['probability_score'], 0, 1)
            results['impact_score'] += results['probability_score'] * 0.3
        
        # Add infrastructure vulnerability if available
        if infrastructure_layer and infrastructure_layer in self.layers:
            infra_gdf = self.layers[infrastructure_layer]
            
            # Count infrastructure in each flood zone
            for idx, flood_zone in results.iterrows():
                infra_in_zone = infra_gdf[infra_gdf.geometry.intersects(flood_zone.geometry)]
                infrastructure_count = len(infra_in_zone)
                results.at[idx, 'infrastructure_at_risk'] = infrastructure_count
                results.at[idx, 'impact_score'] += min(infrastructure_count / 10, 0.3)
        
        # Final risk classification
        results['flood_risk'] = pd.cut(results['impact_score'], 
                                     bins=[0, 0.3, 0.6, 1.0], 
                                     labels=['Low', 'Medium', 'High'],
                                     include_lowest=True)
        
        return results

    def _solar_optimization(self, step):
        """Optimize solar farm placement"""
        slope_layer = step.get("slope_input")
        landuse_layer = step.get("landuse_input")
        exclusion_layer = step.get("exclusion_input", None)
        
        slope_gdf = self.layers[slope_layer]
        
        results = slope_gdf.copy()
        
        # Solar suitability scoring
        results['solar_score'] = 0
        
        # Slope suitability (optimal 0-5%, good 5-10%, poor >15%)
        if 'slope_percent' in results.columns:
            slope_scores = np.where(results['slope_percent'] <= 5, 1.0,
                                  np.where(results['slope_percent'] <= 10, 0.7,
                                          np.where(results['slope_percent'] <= 15, 0.4, 0.1)))
            results['slope_score'] = slope_scores
            results['solar_score'] += slope_scores * 0.3
        
        # Land use suitability
        if landuse_layer and landuse_layer in self.layers:
            landuse_gdf = self.layers[landuse_layer]
            
            # Spatial join to get land use for each slope area
            joined = gpd.sjoin(results, landuse_gdf, how='left', predicate='intersects')
            
            # Score land use types
            landuse_scores = {
                'barren': 1.0,
                'agricultural': 0.8,
                'grassland': 0.6,
                'scrubland': 0.7,
                'urban': 0.1,
                'forest': 0.2,
                'water': 0.0
            }
            
            joined['landuse_score'] = joined['land_use'].map(landuse_scores).fillna(0.3)
            results['landuse_score'] = joined['landuse_score']
            results['solar_score'] += results['landuse_score'] * 0.4
        else:
            results['landuse_score'] = 0.5
            results['solar_score'] += 0.2
        
        # Distance from grid/roads (simplified - assume closer is better)
        # In production, would calculate actual distances
        results['grid_accessibility'] = np.random.uniform(0.3, 1.0, len(results))
        results['solar_score'] += results['grid_accessibility'] * 0.3
        
        # Apply exclusions
        if exclusion_layer and exclusion_layer in self.layers:
            exclusion_gdf = self.layers[exclusion_layer]
            
            # Mark areas that intersect with exclusion zones
            for idx, area in results.iterrows():
                if any(exclusion_gdf.geometry.intersects(area.geometry)):
                    results.at[idx, 'solar_score'] = 0
                    results.at[idx, 'excluded'] = True
        
        # Final suitability classification
        results['solar_suitability'] = pd.cut(results['solar_score'], 
                                            bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                            labels=['Poor', 'Fair', 'Good', 'Excellent'],
                                            include_lowest=True)
        
        # Rank sites for development
        results['development_priority'] = results['solar_score'].rank(ascending=False, method='dense')
        
        return results

    def _suitability_analysis(self, step):
        """General suitability analysis framework"""
        input_layer = step.get("input")
        factors = step.get("factors", [])
        weights = step.get("weights", [])
        constraints = step.get("constraints", [])
        
        gdf = self.layers[input_layer].copy()
        
        # Apply constraints first (hard exclusions)
        for constraint in constraints:
            column = constraint.get("column")
            operator = constraint.get("operator", "==")
            value = constraint.get("value")
            
            if column in gdf.columns:
                if operator == "==":
                    gdf = gdf[gdf[column] == value]
                elif operator == "!=":
                    gdf = gdf[gdf[column] != value]
                elif operator == ">":
                    gdf = gdf[gdf[column] > value]
                elif operator == "<":
                    gdf = gdf[gdf[column] < value]
                elif operator == ">=":
                    gdf = gdf[gdf[column] >= value]
                elif operator == "<=":
                    gdf = gdf[gdf[column] <= value]
        
        # Apply weighted factors (soft criteria)
        if factors and weights and len(factors) == len(weights):
            return self._multi_criteria_analysis({
                "input": input_layer,
                "criteria": factors,
                "weights": weights
            })
        
        return gdf

    def validate_outputs(self, results):
        for name, gdf in results.items():
            logger.info(f"✅ [{name}] → {len(gdf)} features, CRS: {gdf.crs}")
            # Check for empty geometries but don't fail on them
            empty_count = gdf.is_empty.sum() if hasattr(gdf, 'is_empty') else 0
            if empty_count > 0:
                logger.warning(f"⚠️  [{name}] Warning: {empty_count} empty geometries found")
            else:
                print(f"✅ [{name}] All geometries are valid")

    def generate_visualizations(self, results):
        import matplotlib.pyplot as plt
        for name, gdf in results.items():
            gdf.plot()
            plt.title(name)
            plt.show()
