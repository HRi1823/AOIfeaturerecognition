#!/usr/bin/env python3
"""
gee_aoi_vs_aoi_compare.py

Compare two AOIs using Google Earth Engine AlphaEarth (Satellite Embedding) embeddings.

Usage example:
  pip install earthengine-api geopandas shapely numpy pandas faiss-cpu pyproj
  # Authenticate once:
  python -c "import ee; ee.Authenticate(); ee.Initialize()"

  python gee_aoi_vs_aoi_compare.py \
    --aoi1 tumkur.geojson --aoi2 gadag.geojson \
    --year 2024 --cell-size 500 \
    --max-tiles 2500 --scale 10 \
    --output-prefix tumkur_vs_gadag

Notes:
 - AOI inputs may be:
     * a local GeoJSON file path (e.g., tumkur.geojson)
     * a GEE asset id (e.g., users/you/tumkur_asset)
 - The script will build cartesian tiles inside each AOI (EPSG:3857 metric) and
   compute a mean 64-D AlphaEarth embedding per tile using GEE.
 - For safety and to keep this fully automated, the script **requires** that
   the number of tiles per AOI <= --max-tiles (default 2500). Increase cell-size to reduce tiles.
 - If your AOI is large, use a server-side Export approach instead (I can provide that next).
"""

import argparse
import json
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import faiss

# ---------- Helper functions ----------

def init_ee():
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not initialized. Attempting interactive auth...")
        ee.Authenticate()
        ee.Initialize()
    print("Earth Engine initialized.")

def load_aoi_geometry(aoi_source):
    """
    Accepts either:
      - a local GeoJSON filename -> returns geojson dict
      - a GEE asset ID (string starting with 'users/' or 'projects/') -> returns string (we'll treat specially)
    """
    if isinstance(aoi_source, str) and (aoi_source.startswith("users/") or aoi_source.startswith("projects/")):
        return aoi_source  # treat as asset id
    p = Path(aoi_source)
    if p.exists():
        gj = json.loads(p.read_text())
        # Normalize to a polygon/multipolygon geometry dict
        if 'type' in gj and gj['type'] == 'FeatureCollection':
            # take union of features
            return gj
        elif 'type' in gj and (gj['type'] == 'Feature' or gj['type'] == 'Polygon' or gj['type'] == 'MultiPolygon'):
            return gj
        else:
            raise ValueError("Unsupported GeoJSON format in file: " + str(aoi_source))
    # otherwise try parse as raw geojson string
    try:
        gj = json.loads(aoi_source)
        return gj
    except Exception:
        raise ValueError("AOI source not found or invalid: " + str(aoi_source))

def make_grid_features_from_geojson(aoi_geojson, cell_size_m=500):
    """
    Build grid tiles (as GeoJSON features list) that intersect the AOI.
    Uses EPSG:3857 for metric tiling.
    Returns list of GeoJSON feature dicts (Polygon geometries).
    """
    # Convert to GeoDataFrame
    if isinstance(aoi_geojson, str) and (aoi_geojson.startswith("users/") or aoi_geojson.startswith("projects/")):
        raise ValueError("make_grid_features_from_geojson expects a local GeoJSON dict, not GEE asset id.")
    gdf = gpd.GeoDataFrame.from_features([aoi_geojson], crs="EPSG:4326")
    gdf_m = gdf.to_crs(epsg=3857)
    geom_m = gdf_m.geometry.unary_union
    minx, miny, maxx, maxy = geom_m.bounds
    # create grid
    xs = np.arange(int(minx), int(maxx), int(cell_size_m))
    ys = np.arange(int(miny), int(maxy), int(cell_size_m))
    cells = []
    for x in xs:
        for y in ys:
            c = box(x, y, x + cell_size_m, y + cell_size_m)
            inter = c.intersection(geom_m)
            if not inter.is_empty:
                cells.append(inter)
    if len(cells) == 0:
        raise ValueError("No tiles produced. Check AOI geometry or increase cell_size.")
    cell_gdf = gpd.GeoDataFrame({'geometry': cells}, crs="EPSG:3857")
    cell_gdf = cell_gdf.to_crs(epsg=4326)
    features = []
    for idx, geom in enumerate(cell_gdf.geometry):
        feat = {
            "type": "Feature",
            "properties": {"tile_id": int(idx)},
            "geometry": json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry']
        }
        features.append(feat)
    return features

def features_list_to_ee_fc(features):
    ee_feats = []
    for f in features:
        ee_geom = ee.Geometry(f['geometry'])
        ee_feats.append(ee.Feature(ee_geom, f.get('properties', {})))
    return ee.FeatureCollection(ee_feats)

def tile_mean_embedding_mapper(emb_img, scale=10):
    """
    Return a mapper function that will be used in fc.map(...) to compute mean embedding per feature.
    Sets properties A00..A63 on the feature.
    """
    def _map(feat):
        geom = feat.geometry()
        # reduceRegion mean across geometry
        mean_dict = emb_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e9
        )
        # mean_dict may contain A00..A63 keys; attach them as properties
        return feat.set(mean_dict)
    return _map

def ee_fc_get_all_features(fc):
    """
    Utility to safely call getInfo() on a reasonably sized FeatureCollection.
    Returns list of feature dicts (client-side).
    """
    # getInfo might return a dict with 'features'
    info = fc.getInfo()
    if isinstance(info, dict) and 'features' in info:
        return info['features']
    else:
        raise RuntimeError("Unexpected getInfo() return format. Perhaps too large or a server error.")

def extract_vectors_from_features(features, band_count=64):
    """
    Given list of feature dicts (from ee.FeatureCollection.getInfo()), extract vectors A00..A63.
    Returns:
      ids: list of tile ids
      centroids: list of [lon,lat]
      vecs: numpy array N x band_count (float32)
    """
    ids = []
    centroids = []
    rows = []
    for f in features:
        props = f.get('properties', {})
        tid = props.get('tile_id', props.get('id', None))
        # compute centroid from geometry if available
        geom = f.get('geometry')
        cent = None
        try:
            # geometry could be Polygon or MultiPolygon
            coords = geom.get('coordinates', [])
            # approximate centroid by averaging exterior ring coords
            if geom.get('type', None) == 'Polygon':
                ring = coords[0]
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                cent = [sum(xs)/len(xs), sum(ys)/len(ys)]
            else:
                # fallback: 0,0
                cent = [0.0, 0.0]
        except Exception:
            cent = [0.0,0.0]
        vec = []
        for i in range(band_count):
            v = props.get(f"A{str(i).zfill(2)}")
            if v is None:
                # fallback to 0
                vec.append(0.0)
            else:
                vec.append(float(v))
        ids.append(tid)
        centroids.append(cent)
        rows.append(vec)
    arr = np.array(rows, dtype='float32')
    return ids, centroids, arr

def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return X / norms

# ---------- Main pipeline ----------

def main(args):
    init_ee()

    # Load AOIs
    aoi1_src = args.aoi1
    aoi2_src = args.aoi2

    aoi1_obj = load_aoi_geometry(aoi1_src)
    aoi2_obj = load_aoi_geometry(aoi2_src)

    # Build client-side tile grids for AOIs that are local GeoJSONs.
    # For GEE asset AOIs, fetch geometry bounding box and convert to client-side shapely via getInfo.
    def get_local_geojson_for_aoi(aoi_obj):
        if isinstance(aoi_obj, str) and (aoi_obj.startswith("users/") or aoi_obj.startswith("projects/")):
            # load geometry from asset as geojson (careful: getInfo may retrieve a large geometry but usually ok)
            fc = ee.FeatureCollection(aoi_obj)
            geom = fc.geometry().getInfo()  # returns geometry dict
            return {"type": "Feature", "properties": {}, "geometry": geom}
        else:
            # assume local geojson dict or featurecollection
            if isinstance(aoi_obj, dict) and aoi_obj.get('type') == 'FeatureCollection':
                # convert to unary union into a single feature geometry
                geoms = [f['geometry'] for f in aoi_obj.get('features', [])]
                # create an ee.Geometry.MultiPolygon or union locally via geopandas
                gdf = gpd.GeoDataFrame.from_features(aoi_obj, crs="EPSG:4326")
                union = gdf.unary_union
                return json.loads(gpd.GeoSeries([union], crs="EPSG:4326").to_json())['features'][0]
            elif isinstance(aoi_obj, dict) and aoi_obj.get('type') == 'Feature':
                return aoi_obj
            elif isinstance(aoi_obj, dict) and aoi_obj.get('type') in ['Polygon', 'MultiPolygon']:
                return {"type":"Feature","properties":{},"geometry":aoi_obj}
            else:
                raise ValueError("Unsupported AOI object type.")
    print("Preparing AOI geometries and grids...")
    aoi1_feature = get_local_geojson_for_aoi(aoi1_obj)
    aoi2_feature = get_local_geojson_for_aoi(aoi2_obj)

    # build grids
    grid1 = make_grid_features_from_geojson(aoi1_feature, cell_size_m=int(args.cell_size))
    grid2 = make_grid_features_from_geojson(aoi2_feature, cell_size_m=int(args.cell_size))
    print(f"AOI1 -> tiles: {len(grid1)}; AOI2 -> tiles: {len(grid2)}")

    if len(grid1) > args.max_tiles or len(grid2) > args.max_tiles:
        raise SystemExit(f"Tile count exceeds max_tiles ({args.max_tiles}). Increase --cell-size or reduce AOI extent.")

    # Convert grid features to ee.FeatureCollection
    fc1 = features_list_to_ee_fc(grid1)
    fc2 = features_list_to_ee_fc(grid2)

    # Load AlphaEarth embedding image for the requested year
    print("Loading AlphaEarth embedding image (this may take a moment)...")
    emb_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    start = ee.Date.fromYMD(int(args.year), 1, 1)
    end = start.advance(1, 'year')
    emb_img = emb_col.filterDate(start, end).mosaic()

    # Map reduce to add mean embeddings as properties
    print("Mapping reduceRegion(mean) on AOI1 tiles (server-side)...")
    fc1_mapped = fc1.map(tile_mean_embedding_mapper(emb_img, scale=int(args.scale)))
    print("Mapping reduceRegion(mean) on AOI2 tiles (server-side)...")
    fc2_mapped = fc2.map(tile_mean_embedding_mapper(emb_img, scale=int(args.scale)))

    # Get results back (synchronously) — this is safe because we restricted tile count
    print("Retrieving AOI1 tile features (this is synchronous; small tile counts only)...")
    feats1 = ee_fc_get_all_features(fc1_mapped)
    print("Retrieving AOI2 tile features...")
    feats2 = ee_fc_get_all_features(fc2_mapped)

    # Extract vectors, centroids, ids
    print("Extracting vectors...")
    ids1, cents1, X1 = extract_vectors_from_features(feats1, band_count=64)
    ids2, cents2, X2 = extract_vectors_from_features(feats2, band_count=64)

    print(f"AOI1 vectors: {X1.shape}; AOI2 vectors: {X2.shape}")

    # Convert to float32 and normalize for cosine-like comparisons
    X1 = np.array(X1, dtype='float32')
    X2 = np.array(X2, dtype='float32')
    X1n = l2_normalize_rows(X1)
    X2n = l2_normalize_rows(X2)

    # Build FAISS index on AOI2 and query with AOI1
    d = X2n.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X2n)
    k = int(args.k)
    print(f"Searching AOI2 for nearest neighbors of each AOI1 tile (k={k})...")
    D, I = index.search(X1n, k)

    # D: distances (L2 on normalized vectors). Convert to cosine similarity if desired:
    # For normalized vectors, cosine sim = 1 - 0.5 * L2^2 (not necessary here). We'll report both.
    results_rows = []
    for i in range(X1n.shape[0]):
        for nn in range(k):
            idx2 = int(I[i, nn])
            dist = float(D[i, nn])
            # compute cosine similarity
            cos_sim = float(np.dot(X1n[i], X2n[idx2]))
            row = {
                'aoi1_tile_id': ids1[i],
                'aoi1_centroid_lon': float(cents1[i][0]),
                'aoi1_centroid_lat': float(cents1[i][1]),
                'rank': nn + 1,
                'aoi2_tile_id': ids2[idx2],
                'aoi2_centroid_lon': float(cents2[idx2][0]),
                'aoi2_centroid_lat': float(cents2[idx2][1]),
                'l2_dist': dist,
                'cosine_sim': cos_sim
            }
            results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    prefix = args.output_prefix
    csv_path = f"{prefix}_tile_nn_results.csv"
    json_path = f"{prefix}_summary.json"
    results_df.to_csv(csv_path, index=False)
    print("Tile-wise nearest-neighbor results written to:", csv_path)

    # Summaries: for each aoi1 tile, take nearest (rank=1) distance
    rank1 = results_df[results_df['rank'] == 1].copy()
    l2_arr = rank1['l2_dist'].values
    cos_arr = rank1['cosine_sim'].values

    summary = {
        'aoi1_tile_count': int(X1n.shape[0]),
        'aoi2_tile_count': int(X2n.shape[0]),
        'l2_mean': float(np.mean(l2_arr)),
        'l2_median': float(np.median(l2_arr)),
        'l2_std': float(np.std(l2_arr)),
        'cos_mean': float(np.mean(cos_arr)),
        'cos_median': float(np.median(cos_arr)),
        'cos_std': float(np.std(cos_arr)),
        # percentiles
        'l2_25': float(np.percentile(l2_arr, 25)),
        'l2_75': float(np.percentile(l2_arr, 75)),
        'cos_25': float(np.percentile(cos_arr, 25)),
        'cos_75': float(np.percentile(cos_arr, 75))
    }

    Path(json_path).write_text(json.dumps(summary, indent=2))
    print("Summary written to:", json_path)

    # Print quick stats
    print("\n--- Summary ---")
    print("AOI1 tiles:", summary['aoi1_tile_count'], "AOI2 tiles:", summary['aoi2_tile_count'])
    print("L2 mean / median:", summary['l2_mean'], "/", summary['l2_median'])
    print("Cosine mean / median:", summary['cos_mean'], "/", summary['cos_median'])
    print("----------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two AOIs using GEE AlphaEarth embeddings (tile-level).")
    parser.add_argument("--aoi1", required=True, help="AOI1: local GeoJSON file path or GEE asset id (users/... )")
    parser.add_argument("--aoi2", required=True, help="AOI2: local GeoJSON file path or GEE asset id")
    parser.add_argument("--year", default=2024, help="Year to use from AlphaEarth embeddings")
    parser.add_argument("--cell-size", default=500, help="Grid cell size in meters (use 200-1000 depending scale)")
    parser.add_argument("--max-tiles", default=2500, type=int, help="Maximum tiles allowed per AOI for synchronous getInfo.")
    parser.add_argument("--scale", default=10, help="GEE sampling scale (m) for reduceRegion")
    parser.add_argument("--k", default=1, help="Nearest neighbors per AOI1 tile to retrieve")
    parser.add_argument("--output-prefix", default="aoi_compare", help="Prefix for output files")
    args = parser.parse_args()
    main(args)
