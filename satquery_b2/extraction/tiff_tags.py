"""
SatQuery AI - B2 GeoTIFF Tag and Key Constants
==============================================
Standard TIFF/GeoTIFF tags, GeoKey directory structures, and EPSG resolution tables.
"""

from typing import Any, Dict, Optional, Tuple

# Standard Baseline TIFF Tag IDs
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257  # Height
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC_INTERPRETATION = 262
TAG_IMAGE_DESCRIPTION = 270
TAG_STRIP_OFFSETS = 273
TAG_ORIENTATION = 274
TAG_SAMPLES_PER_PIXEL = 277  # Band count
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIGURATION = 284
TAG_SOFTWARE = 305
TAG_DATE_TIME = 306
TAG_ARTIST = 315
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_SAMPLE_FORMAT = 339

# GeoTIFF Extension Tag IDs
TAG_MODEL_PIXEL_SCALE = 33550      # (ScaleX, ScaleY, ScaleZ)
TAG_MODEL_TIEPOINT = 33922         # (I, J, K, X, Y, Z)
TAG_MODEL_TRANSFORMATION = 34264   # 4x4 matrix
TAG_GEO_KEY_DIRECTORY = 34735      # Key Directory Version, Key Revision, Minor Revision, NumKeys, ...
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737
TAG_GDAL_METADATA = 42112
TAG_GDAL_NODATA = 42113

# GeoKey IDs (Inside GeoKeyDirectoryTag)
GEOKEY_GT_MODEL_TYPE = 1024         # ModelType (1=Projected, 2=Geographic, 3=Geocentric)
GEOKEY_GT_RASTER_TYPE = 1025        # RasterType (1=PixelIsArea, 2=PixelIsPoint)
GEOKEY_GT_CITATION = 1026
GEOKEY_GEOGRAPHIC_TYPE = 2048       # Geographic CS (e.g. 4326 for WGS84)
GEOKEY_GEOG_CITATION = 2049
GEOKEY_GEOG_GEODETIC_DATUM = 2050
GEOKEY_PROJECTED_CS_TYPE = 3072     # Projected CS (e.g. 32643 for UTM zone 43N)
GEOKEY_PCS_CITATION = 3073
GEOKEY_PROJECTION = 3074
GEOKEY_PROJ_COORD_TRANS_CODE = 3075
GEOKEY_PROJ_LINEAR_UNITS = 3076

# Common EPSG mappings
STANDARD_EPSG_NAMES: Dict[int, str] = {
    4326: "WGS 84 (Geographic Lat/Lon)",
    3857: "WGS 84 / Pseudo-Mercator (Web Mercator)",
    32601: "WGS 84 / UTM zone 1N",
    32632: "WGS 84 / UTM zone 32N",
    32633: "WGS 84 / UTM zone 33N",
    32643: "WGS 84 / UTM zone 43N",
    32644: "WGS 84 / UTM zone 44N",
    32732: "WGS 84 / UTM zone 32S",
}


def parse_geokey_directory(raw_keys: Tuple[int, ...]) -> Dict[int, Any]:
    """
    Parses a raw GeoKeyDirectory tuple into a key-value dictionary.
    GeoKeyDirectory header is 4 unsigned shorts: (KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys).
    Each key entry is 4 unsigned shorts: (KeyID, TIFFTagLocation, Count, Value_Offset).
    """
    if not raw_keys or len(raw_keys) < 4:
        return {}

    parsed_keys: Dict[int, Any] = {}
    num_keys = raw_keys[3]

    for i in range(num_keys):
        offset = 4 + (i * 4)
        if offset + 3 >= len(raw_keys):
            break
        key_id = raw_keys[offset]
        tag_loc = raw_keys[offset + 1]
        count = raw_keys[offset + 2]
        val_offset = raw_keys[offset + 3]

        if tag_loc == 0:
            # Value is stored directly in val_offset
            parsed_keys[key_id] = val_offset
        else:
            # Stored in external tag
            parsed_keys[key_id] = (tag_loc, count, val_offset)

    return parsed_keys


def resolve_epsg_from_geokeys(geokeys: Dict[int, Any]) -> Tuple[Optional[int], Optional[str]]:
    """
    Extracts EPSG code and CRS string from parsed GeoKeys.
    Returns (epsg_code, crs_string).
    """
    # Check Projected CS first
    pcs = geokeys.get(GEOKEY_PROJECTED_CS_TYPE)
    if pcs and isinstance(pcs, int) and pcs > 0 and pcs != 32767:
        return pcs, f"EPSG:{pcs}"

    # Check Geographic CS
    gcs = geokeys.get(GEOKEY_GEOGRAPHIC_TYPE)
    if gcs and isinstance(gcs, int) and gcs > 0 and gcs != 32767:
        return gcs, f"EPSG:{gcs}"

    return None, None
