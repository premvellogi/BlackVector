"""
SatQuery AI - B2 Raster Metadata Extraction Layer
==================================================
Reads geospatial raster files (GeoTIFF/TIFF/PNG/JPEG) and constructs canonical
OntologyImage instances with zero unnecessary array copies.
"""

from __future__ import annotations

import os
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from satquery_b2.extraction.tiff_tags import (
    TAG_DATE_TIME,
    TAG_GDAL_METADATA,
    TAG_GEO_KEY_DIRECTORY,
    TAG_IMAGE_DESCRIPTION,
    TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
    TAG_SAMPLES_PER_PIXEL,
    parse_geokey_directory,
    resolve_epsg_from_geokeys,
)
from satquery_b2.ontology.schema import (
    AffineTransform,
    BandInfo,
    FileFormat,
    Modality,
    OntologyImage,
    SpatialBounds,
)


class RasterExtractionError(Exception):
    """Base exception for raster extraction failures."""
    pass


class UnreadableRasterError(RasterExtractionError):
    """Raised when a raster file cannot be opened or read."""
    pass


class CorruptedRasterError(RasterExtractionError):
    """Raised when a raster header or data stream is corrupted."""
    pass


class UnsupportedRasterFormatError(RasterExtractionError):
    """Raised when the file format is not a supported raster format."""
    pass


def _extract_via_rasterio(
    file_path: str,
    source_id: str,
    image_id: str,
    dataset_name: Optional[str] = None,
    default_modality: Optional[Modality] = None,
) -> Optional[OntologyImage]:
    """Attempts metadata extraction using rasterio if available in the environment."""
    try:
        import rasterio
    except ImportError:
        return None

    try:
        with rasterio.open(file_path) as src:
            width = src.width
            height = src.height
            band_count = src.count
            dtypes = src.dtypes

            # CRS & EPSG
            crs_str = str(src.crs) if src.crs else None
            epsg = src.crs.to_epsg() if src.crs else None

            # Spatial resolution
            res_x, res_y = src.res if src.res else (None, None)
            spatial_res = float(abs(res_x)) if res_x is not None and abs(res_x) > 0 else None

            # Affine transform
            t = src.transform
            affine = AffineTransform(a=t.a, b=t.b, c=t.c, d=t.d, e=t.e, f=t.f) if t else None

            # Spatial Bounds
            b = src.bounds
            spatial_bounds = SpatialBounds(
                min_x=float(b.left),
                min_y=float(b.bottom),
                max_x=float(b.right),
                max_y=float(b.top),
                crs=crs_str,
            ) if b else None

            # Bands
            bands: List[BandInfo] = []
            band_descs = src.descriptions or ()
            for idx in range(1, band_count + 1):
                dtype_name = dtypes[idx - 1] if idx - 1 < len(dtypes) else "uint8"
                desc = band_descs[idx - 1] if idx - 1 < len(band_descs) else None
                bands.append(BandInfo(band_index=idx, band_id=desc, description=desc, dtype=dtype_name))

            # Tags and metadata
            tags = src.tags() or {}
            sensor = tags.get("SENSOR") or tags.get("PLATFORM") or tags.get("satellite")
            platform = tags.get("PLATFORM") or tags.get("MISSION")
            acq_time = tags.get("DATETIME") or tags.get("ACQUISITION_DATE") or tags.get("TIFFTAG_DATETIME")

            # Cloud cover
            cloud_cover: Optional[float] = None
            if "CLOUD_COVER" in tags:
                try:
                    cloud_cover = float(tags["CLOUD_COVER"])
                except (ValueError, TypeError):
                    pass

            # Modality detection
            modality = default_modality or Modality.OTHER
            if "MODALITY" in tags:
                try:
                    modality = Modality.from_str(tags["MODALITY"])
                except Exception:
                    pass
            elif band_count in (3, 4) and default_modality is None:
                modality = Modality.OPTICAL
            elif band_count in (1, 2) and ("VV" in str(band_descs) or "VH" in str(band_descs)):
                modality = Modality.SAR

            return OntologyImage(
                image_id=image_id,
                source_id=source_id,
                file_format=FileFormat.GEOTIFF.value if "GTiff" in src.driver else FileFormat.TIFF.value,
                modality=modality,
                width=width,
                height=height,
                band_count=band_count,
                sensor=sensor,
                platform=platform,
                acquisition_time=acq_time,
                crs=crs_str,
                epsg=epsg,
                spatial_resolution_m=spatial_res,
                bands=bands,
                cloud_cover_percentage=cloud_cover,
                spatial_bounds=spatial_bounds,
                affine_transform=affine,
                dataset_name=dataset_name,
                metadata_quality={"source": "rasterio", "is_complete": bool(crs_str and epsg)},
                extra_metadata={"driver": src.driver, "tags": tags},
            )
    except Exception as e:
        if isinstance(e, (UnreadableRasterError, CorruptedRasterError)):
            raise
        # Fall back to pure TIFF parser on other rasterio errors
        return None


def _extract_via_pil_and_tags(
    file_path: str,
    source_id: str,
    image_id: str,
    dataset_name: Optional[str] = None,
    default_modality: Optional[Modality] = None,
) -> OntologyImage:
    """Robust zero-dependency GeoTIFF / TIFF / PNG / JPEG metadata parser using Pillow."""
    from PIL import Image

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = (img.format or "TIFF").upper()

            # Handle non-TIFF formats (PNG, JPEG)
            if img_format not in ("TIFF", "TIF"):
                mode = img.mode
                band_count = len(mode) if mode in ("RGB", "RGBA", "CMYK") else 1
                bands = [BandInfo(band_index=i + 1, dtype="uint8") for i in range(band_count)]
                modality = default_modality or (Modality.OPTICAL if band_count >= 3 else Modality.OTHER)

                return OntologyImage(
                    image_id=image_id,
                    source_id=source_id,
                    file_format=img_format,
                    modality=modality,
                    width=width,
                    height=height,
                    band_count=band_count,
                    bands=bands,
                    dataset_name=dataset_name,
                    metadata_quality={"source": "pil_standard", "is_complete": False, "has_georeference": False},
                )

            # Extract TIFF IFD Tags
            tag_dict: Dict[int, Any] = {}
            if hasattr(img, "tag_v2"):
                tag_dict = dict(img.tag_v2)
            elif hasattr(img, "tag"):
                tag_dict = dict(img.tag)

            # Band count (SamplesPerPixel or channel count)
            band_count = tag_dict.get(TAG_SAMPLES_PER_PIXEL, len(img.getbands()) if hasattr(img, "getbands") else 1)
            if isinstance(band_count, tuple):
                band_count = band_count[0]

            # Bands metadata
            bands = []
            for b_idx in range(1, band_count + 1):
                bands.append(BandInfo(band_index=b_idx, dtype="uint8"))

            # GeoTIFF Tags: ModelPixelScale & ModelTiepoint
            pixel_scale = tag_dict.get(TAG_MODEL_PIXEL_SCALE)
            tiepoint = tag_dict.get(TAG_MODEL_TIEPOINT)
            geokey_dir = tag_dict.get(TAG_GEO_KEY_DIRECTORY)

            affine: Optional[AffineTransform] = None
            spatial_bounds: Optional[SpatialBounds] = None
            spatial_res: Optional[float] = None
            epsg: Optional[int] = None
            crs_str: Optional[str] = None

            # Calculate Affine & Bounds if GeoTIFF tags exist
            if pixel_scale and tiepoint and len(pixel_scale) >= 2 and len(tiepoint) >= 6:
                sx, sy = float(pixel_scale[0]), float(pixel_scale[1])
                # tiepoint: (i, j, k, x, y, z) -> top-left raster coords (i,j) map to world (x,y)
                i, j, _, x, y, _ = tiepoint[:6]
                origin_x = x - (i * sx)
                origin_y = y + (j * sy)

                affine = AffineTransform(
                    a=sx,
                    b=0.0,
                    c=origin_x,
                    d=0.0,
                    e=-sy,
                    f=origin_y,
                )
                spatial_res = float(sx)

                # Bounding box
                min_x = origin_x
                max_x = origin_x + (width * sx)
                max_y = origin_y
                min_y = origin_y - (height * sy)

                # Check GeoKeys for CRS/EPSG
                if geokey_dir:
                    parsed_geokeys = parse_geokey_directory(geokey_dir)
                    epsg, crs_str = resolve_epsg_from_geokeys(parsed_geokeys)

                spatial_bounds = SpatialBounds(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    crs=crs_str,
                )

            # Metadata tags: DateTime, Sensor, Cloud Cover
            acq_time = tag_dict.get(TAG_DATE_TIME)
            if isinstance(acq_time, tuple):
                acq_time = acq_time[0] if acq_time else None
            if isinstance(acq_time, bytes):
                acq_time = acq_time.decode("utf-8", errors="ignore").strip("\x00")

            # Check GDAL metadata tag (42112)
            gdal_meta = tag_dict.get(TAG_GDAL_METADATA)
            sensor = None
            platform = None
            cloud_cover = None
            extra_meta: Dict[str, Any] = {}

            if gdal_meta:
                if isinstance(gdal_meta, bytes):
                    gdal_meta_str = gdal_meta.decode("utf-8", errors="ignore")
                elif isinstance(gdal_meta, tuple):
                    gdal_meta_str = "".join(str(x) for x in gdal_meta)
                else:
                    gdal_meta_str = str(gdal_meta)

                try:
                    root = ET.fromstring(gdal_meta_str)
                    for item in root.findall(".//Item"):
                        name = item.attrib.get("name", "").upper()
                        val = item.text
                        extra_meta[name] = val
                        if "SENSOR" in name or "PLATFORM" in name:
                            sensor = val
                        elif "CLOUD" in name:
                            try:
                                cloud_cover = float(val)
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    extra_meta["raw_gdal_metadata"] = gdal_meta_str

            # Modality resolution
            modality = default_modality or Modality.OTHER
            if "MODALITY" in extra_meta:
                try:
                    modality = Modality.from_str(extra_meta["MODALITY"])
                except Exception:
                    pass
            elif default_modality is None:
                if band_count in (3, 4):
                    modality = Modality.OPTICAL
                elif band_count in (1, 2) and (sensor and "SAR" in sensor.upper()):
                    modality = Modality.SAR

            return OntologyImage(
                image_id=image_id,
                source_id=source_id,
                file_format=FileFormat.GEOTIFF.value if pixel_scale else FileFormat.TIFF.value,
                modality=modality,
                width=width,
                height=height,
                band_count=band_count,
                sensor=sensor,
                platform=platform,
                acquisition_time=acq_time,
                crs=crs_str,
                epsg=epsg,
                spatial_resolution_m=spatial_res,
                bands=bands,
                cloud_cover_percentage=cloud_cover,
                spatial_bounds=spatial_bounds,
                affine_transform=affine,
                dataset_name=dataset_name,
                metadata_quality={"source": "pil_geotiff_parser", "is_complete": bool(crs_str and epsg)},
                extra_metadata=extra_meta,
            )

    except Exception as e:
        if isinstance(e, RasterExtractionError):
            raise
        raise CorruptedRasterError(f"Failed to read raster file '{file_path}': {str(e)}")


def extract_raster_metadata(
    file_path: Union[str, Path],
    source_id: Optional[str] = None,
    image_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    default_modality: Optional[Modality] = None,
) -> OntologyImage:
    """
    Central extraction entry point for converting any geospatial raster into an `OntologyImage`.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the image file.
    source_id : str, optional
        Source identifier (defaults to file path).
    image_id : str, optional
        Unique image ID (defaults to generated UUID4).
    dataset_name : str, optional
        Dataset or benchmark name.
    default_modality : Modality, optional
        Fallback modality if not detectable from metadata.

    Returns
    -------
    OntologyImage
        Canonical parsed and validated ontology image.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise UnreadableRasterError(f"Raster file does not exist or is not a regular file: {file_path}")

    # Check file extension
    ext = path.suffix.lower()
    if ext not in (".tif", ".tiff", ".geotiff", ".gtif", ".png", ".jpg", ".jpeg"):
        raise UnsupportedRasterFormatError(f"Unsupported file format extension '{ext}' for file {file_path}")

    # Generate identifiers
    eff_source_id = source_id or str(path)
    eff_image_id = image_id or f"img-{uuid.uuid4().hex[:12]}"

    # Try rasterio first if available
    result = _extract_via_rasterio(
        file_path=str(path),
        source_id=eff_source_id,
        image_id=eff_image_id,
        dataset_name=dataset_name,
        default_modality=default_modality,
    )
    if result is not None:
        return result

    # Fall back to PIL + GeoTIFF tags parser
    return _extract_via_pil_and_tags(
        file_path=str(path),
        source_id=eff_source_id,
        image_id=eff_image_id,
        dataset_name=dataset_name,
        default_modality=default_modality,
    )
