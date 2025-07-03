# gis_engine/crs_utils.py

def harmonize_crs(gdf, target_crs="EPSG:4326"):
    if gdf.crs is None:
        print("[CRSUtils] Warning: Input has no CRS.")
        return gdf.set_crs(target_crs)
    
    if gdf.crs.to_string() != target_crs:
        print(f"[CRSUtils] Reprojecting from {gdf.crs} to {target_crs}")
        return gdf.to_crs(target_crs)
    
    print("[CRSUtils] CRS already matches target.")
    return gdf
