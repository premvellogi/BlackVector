"""
SatQuery AI - B2 Metadata Extraction Package
"""

from satquery_b2.extraction.raster_metadata import (
    CorruptedRasterError,
    RasterExtractionError,
    UnreadableRasterError,
    UnsupportedRasterFormatError,
    extract_raster_metadata,
)
from satquery_b2.extraction.tiff_tags import (
    parse_geokey_directory,
    resolve_epsg_from_geokeys,
)

__all__ = [
    "extract_raster_metadata",
    "RasterExtractionError",
    "UnreadableRasterError",
    "CorruptedRasterError",
    "UnsupportedRasterFormatError",
    "parse_geokey_directory",
    "resolve_epsg_from_geokeys",
]
