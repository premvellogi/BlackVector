"""
SatQuery AI - B2 Canonical Image Ontology Schema
=================================================
Defines the canonical `OntologyImage` and associated structures so downstream
SatQuery AI components consume a unified, validated representation rather than
repeatedly parsing raw satellite raster metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from satquery_b2.ontology.exceptions import (
    InvalidMetadataError,
    ModalityMismatchError,
    SchemaSerializationError,
)


class Modality(str, Enum):
    """Supported imagery modalities in SatQuery AI."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    ELEVATION = "elevation"
    OTHER = "other"

    @classmethod
    def from_str(cls, value: Union[str, Modality]) -> Modality:
        if isinstance(value, Modality):
            return value
        if not isinstance(value, str):
            raise InvalidMetadataError("modality", f"Modality must be a string or Modality enum, got {type(value)}")
        val_lower = value.strip().lower()
        for member in cls:
            if member.value == val_lower:
                return member
        raise InvalidMetadataError("modality", f"Unsupported modality '{value}'. Allowed: {[m.value for m in cls]}", value)


class FileFormat(str, Enum):
    """Recognized raster and benchmark image formats."""
    GEOTIFF = "GEOTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    OTHER = "OTHER"

    @classmethod
    def from_str(cls, value: Union[str, FileFormat]) -> FileFormat:
        if isinstance(value, FileFormat):
            return value
        if not isinstance(value, str):
            raise InvalidMetadataError("file_format", f"File format must be a string, got {type(value)}")
        val_upper = value.strip().upper()
        if val_upper in ("GEOTIFF", "GEO_TIFF", "GTIF", "TIF", "TIFF"):
            return cls.GEOTIFF if "GEO" in val_upper else cls.TIFF
        for member in cls:
            if member.value == val_upper:
                return member
        return cls.OTHER


@dataclass(frozen=True)
class SpatialBounds:
    """Geographic or projected bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: Optional[str] = None

    def __post_init__(self) -> None:
        if self.min_x > self.max_x:
            raise InvalidMetadataError("spatial_bounds.min_x", f"min_x ({self.min_x}) cannot exceed max_x ({self.max_x})")
        if self.min_y > self.max_y:
            raise InvalidMetadataError("spatial_bounds.min_y", f"min_y ({self.min_y}) cannot exceed max_y ({self.max_y})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_x": float(self.min_x),
            "min_y": float(self.min_y),
            "max_x": float(self.max_x),
            "max_y": float(self.max_y),
            "crs": self.crs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialBounds:
        try:
            return cls(
                min_x=float(data["min_x"]),
                min_y=float(data["min_y"]),
                max_x=float(data["max_x"]),
                max_y=float(data["max_y"]),
                crs=data.get("crs"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise InvalidMetadataError("spatial_bounds", f"Failed to parse SpatialBounds: {str(e)}", data)

    def calculate_intersection_iou(self, other: SpatialBounds) -> float:
        """Calculate spatial intersection over union (IoU) between two bounding boxes."""
        inter_min_x = max(self.min_x, other.min_x)
        inter_min_y = max(self.min_y, other.min_y)
        inter_max_x = min(self.max_x, other.max_x)
        inter_max_y = min(self.max_y, other.max_y)

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_self = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        area_other = (other.max_x - other.min_x) * (other.max_y - other.min_y)
        union_area = area_self + area_other - inter_area

        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)


@dataclass(frozen=True)
class AffineTransform:
    """Affine transformation matrix parameters (GDAL/Rasterio convention: a, b, c, d, e, f)."""
    a: float  # pixel width (x-resolution)
    b: float  # row rotation
    c: float  # x-origin (top-left x)
    d: float  # column rotation
    e: float  # pixel height (y-resolution, typically negative)
    f: float  # y-origin (top-left y)

    def to_dict(self) -> Dict[str, float]:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
            "d": float(self.d),
            "e": float(self.e),
            "f": float(self.f),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AffineTransform:
        try:
            return cls(
                a=float(data["a"]),
                b=float(data["b"]),
                c=float(data["c"]),
                d=float(data["d"]),
                e=float(data["e"]),
                f=float(data["f"]),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise InvalidMetadataError("affine_transform", f"Failed to parse AffineTransform: {str(e)}", data)


@dataclass
class BandInfo:
    """Band-level metadata."""
    band_index: int  # 1-indexed according to geospatial conventions
    band_id: Optional[str] = None  # e.g., 'B02', 'B03', 'B04', 'B08', 'VV', 'VH'
    wavelength_nm: Optional[float] = None  # Center wavelength if optical/multispectral
    description: Optional[str] = None
    dtype: str = "uint8"

    def __post_init__(self) -> None:
        if self.band_index < 1:
            raise InvalidMetadataError("band_index", f"Band index must be >= 1, got {self.band_index}")
        if self.wavelength_nm is not None and self.wavelength_nm <= 0:
            raise InvalidMetadataError("wavelength_nm", f"Wavelength must be positive, got {self.wavelength_nm}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_index": self.band_index,
            "band_id": self.band_id,
            "wavelength_nm": self.wavelength_nm,
            "description": self.description,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BandInfo:
        try:
            return cls(
                band_index=int(data["band_index"]),
                band_id=data.get("band_id"),
                wavelength_nm=float(data["wavelength_nm"]) if data.get("wavelength_nm") is not None else None,
                description=data.get("description"),
                dtype=str(data.get("dtype", "uint8")),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise InvalidMetadataError("band_info", f"Failed to parse BandInfo: {str(e)}", data)


@dataclass
class OntologyImage:
    """
    Canonical image ontology representation for SatQuery AI.
    Every downstream model or pipeline consumes this schema instead of
    re-parsing raw satellite raster headers.
    """
    image_id: str
    source_id: str
    file_format: str
    modality: Modality
    width: int
    height: int
    band_count: int
    sensor: Optional[str] = None
    platform: Optional[str] = None
    acquisition_time: Optional[str] = None
    crs: Optional[str] = None
    epsg: Optional[int] = None
    spatial_resolution_m: Optional[float] = None
    bands: List[BandInfo] = field(default_factory=list)
    cloud_cover_percentage: Optional[float] = None
    spatial_bounds: Optional[SpatialBounds] = None
    affine_transform: Optional[AffineTransform] = None
    dataset_name: Optional[str] = None
    metadata_quality: Dict[str, Any] = field(default_factory=lambda: {"is_complete": True, "source": "extracted"})
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate image_id and source_id
        if not self.image_id or not isinstance(self.image_id, str):
            raise InvalidMetadataError("image_id", "Image ID must be a non-empty string", self.image_id)
        if not self.source_id or not isinstance(self.source_id, str):
            raise InvalidMetadataError("source_id", "Source ID must be a non-empty string", self.source_id)

        # Validate modality
        if not isinstance(self.modality, Modality):
            self.modality = Modality.from_str(self.modality)

        # Validate file format
        if not self.file_format or not isinstance(self.file_format, str):
            raise InvalidMetadataError("file_format", "File format must be a non-empty string", self.file_format)

        # Validate dimensions
        if not isinstance(self.width, int) or self.width <= 0:
            raise InvalidMetadataError("width", f"Width must be a positive integer, got {self.width}")
        if not isinstance(self.height, int) or self.height <= 0:
            raise InvalidMetadataError("height", f"Height must be a positive integer, got {self.height}")

        # Validate band count
        if not isinstance(self.band_count, int) or self.band_count <= 0:
            raise InvalidMetadataError("band_count", f"Band count must be a positive integer, got {self.band_count}")

        # Validate bands list if present
        if self.bands:
            if len(self.bands) != self.band_count:
                raise InvalidMetadataError(
                    "bands",
                    f"Number of BandInfo entries ({len(self.bands)}) does not match band_count ({self.band_count})",
                )

        # Validate spatial resolution
        if self.spatial_resolution_m is not None:
            if not isinstance(self.spatial_resolution_m, (int, float)) or self.spatial_resolution_m <= 0:
                raise InvalidMetadataError(
                    "spatial_resolution_m",
                    f"Spatial resolution must be a positive float, got {self.spatial_resolution_m}",
                )
            self.spatial_resolution_m = float(self.spatial_resolution_m)

        # Validate EPSG
        if self.epsg is not None:
            if not isinstance(self.epsg, int) or self.epsg <= 0:
                raise InvalidMetadataError("epsg", f"EPSG code must be a positive integer, got {self.epsg}")

        # Validate CRS string
        if self.crs is not None:
            if not isinstance(self.crs, str) or not self.crs.strip():
                raise InvalidMetadataError("crs", "CRS representation cannot be empty or blank string")

        # Validate cloud cover
        if self.cloud_cover_percentage is not None:
            if not isinstance(self.cloud_cover_percentage, (int, float)):
                raise InvalidMetadataError("cloud_cover_percentage", "Cloud cover must be numeric")
            if not (0.0 <= float(self.cloud_cover_percentage) <= 100.0):
                raise InvalidMetadataError(
                    "cloud_cover_percentage",
                    f"Cloud cover must be between 0.0 and 100.0 inclusive, got {self.cloud_cover_percentage}",
                )
            self.cloud_cover_percentage = float(self.cloud_cover_percentage)

    def is_optical(self) -> bool:
        """Check if image is optical/multispectral/hyperspectral."""
        return self.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.HYPERSPECTRAL)

    def is_sar(self) -> bool:
        """Check if image is SAR."""
        return self.modality == Modality.SAR

    def to_dict(self) -> Dict[str, Any]:
        """Serialize OntologyImage to a JSON-compatible Python dictionary."""
        return {
            "image_id": self.image_id,
            "source_id": self.source_id,
            "file_format": self.file_format,
            "modality": self.modality.value,
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "sensor": self.sensor,
            "platform": self.platform,
            "acquisition_time": self.acquisition_time,
            "crs": self.crs,
            "epsg": self.epsg,
            "spatial_resolution_m": self.spatial_resolution_m,
            "bands": [b.to_dict() for b in self.bands],
            "cloud_cover_percentage": self.cloud_cover_percentage,
            "spatial_bounds": self.spatial_bounds.to_dict() if self.spatial_bounds else None,
            "affine_transform": self.affine_transform.to_dict() if self.affine_transform else None,
            "dataset_name": self.dataset_name,
            "metadata_quality": self.metadata_quality,
            "extra_metadata": self.extra_metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize OntologyImage to a JSON formatted string."""
        try:
            return json.dumps(self.to_dict(), indent=indent, default=str)
        except Exception as e:
            raise SchemaSerializationError(f"Failed to serialize OntologyImage to JSON: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OntologyImage:
        """Construct an OntologyImage instance from a dictionary."""
        if not isinstance(data, dict):
            raise SchemaSerializationError(f"Expected dictionary for OntologyImage, got {type(data)}")

        try:
            # Parse sub-objects
            bands_data = data.get("bands", [])
            bands = [BandInfo.from_dict(b) if isinstance(b, dict) else b for b in bands_data]

            bounds_data = data.get("spatial_bounds")
            spatial_bounds = SpatialBounds.from_dict(bounds_data) if bounds_data else None

            affine_data = data.get("affine_transform")
            affine_transform = AffineTransform.from_dict(affine_data) if affine_data else None

            modality_val = data.get("modality", Modality.OTHER.value)
            modality = Modality.from_str(modality_val)

            return cls(
                image_id=str(data["image_id"]),
                source_id=str(data["source_id"]),
                file_format=str(data.get("file_format", "GEOTIFF")),
                modality=modality,
                width=int(data["width"]),
                height=int(data["height"]),
                band_count=int(data["band_count"]),
                sensor=data.get("sensor"),
                platform=data.get("platform"),
                acquisition_time=data.get("acquisition_time"),
                crs=data.get("crs"),
                epsg=int(data["epsg"]) if data.get("epsg") is not None else None,
                spatial_resolution_m=float(data["spatial_resolution_m"]) if data.get("spatial_resolution_m") is not None else None,
                bands=bands,
                cloud_cover_percentage=float(data["cloud_cover_percentage"]) if data.get("cloud_cover_percentage") is not None else None,
                spatial_bounds=spatial_bounds,
                affine_transform=affine_transform,
                dataset_name=data.get("dataset_name"),
                metadata_quality=data.get("metadata_quality", {}),
                extra_metadata=data.get("extra_metadata", {}),
            )
        except KeyError as e:
            raise InvalidMetadataError(str(e), f"Missing required field in OntologyImage dictionary: {str(e)}")
        except Exception as e:
            if isinstance(e, InvalidMetadataError):
                raise
            raise SchemaSerializationError(f"Failed to deserialize OntologyImage from dictionary: {str(e)}")

    @classmethod
    def from_json(cls, json_str: str) -> OntologyImage:
        """Construct an OntologyImage instance from a JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise SchemaSerializationError(f"Malformed JSON string: {str(e)}")


class PairType(str, Enum):
    """Supported pair configuration types in SatQuery AI."""
    OPTICAL_SAR = "OPTICAL_SAR"
    OPTICAL_TEMPORAL = "OPTICAL_TEMPORAL"
    SAR_TEMPORAL = "SAR_TEMPORAL"

    @classmethod
    def from_str(cls, value: Union[str, PairType]) -> PairType:
        if isinstance(value, PairType):
            return value
        if not isinstance(value, str):
            raise InvalidMetadataError("pair_type", f"PairType must be a string, got {type(value)}")
        val_upper = value.strip().upper()
        for member in cls:
            if member.value == val_upper or member.name == val_upper:
                return member
        raise InvalidMetadataError("pair_type", f"Unsupported PairType '{value}'. Allowed: {[m.value for m in cls]}")


class PairedOntologyImage:
    """
    Container for paired cross-modal (Optical + SAR) or bi-temporal (Optical+Optical / SAR+SAR) imagery.
    Preserves canonical ontology models with type-safe cross-modality and temporal verification.
    """
    def __init__(
        self,
        pair_id: str,
        primary_image: Optional[OntologyImage] = None,
        secondary_image: Optional[OntologyImage] = None,
        pair_type: Union[PairType, str] = PairType.OPTICAL_SAR,
        optical_image: Optional[OntologyImage] = None,
        sar_image: Optional[OntologyImage] = None,
        co_registration_status: Optional[str] = "unverified",
        spatial_overlap_iou: Optional[float] = None,
        temporal_baseline_days: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pair_id = str(pair_id) if pair_id else ""
        if not self.pair_id:
            raise InvalidMetadataError("pair_id", "Pair ID must be a non-empty string")

        # Resolve images: support both (primary, secondary) and legacy (optical_image, sar_image)
        if optical_image is not None and sar_image is not None:
            self.primary_image = optical_image
            self.secondary_image = sar_image
            self.pair_type = PairType.OPTICAL_SAR
        elif primary_image is not None and secondary_image is not None:
            self.primary_image = primary_image
            self.secondary_image = secondary_image
            self.pair_type = PairType.from_str(pair_type)
        else:
            raise InvalidMetadataError(
                "paired_images",
                "Must provide either (primary_image, secondary_image) or (optical_image, sar_image)"
            )

        self.co_registration_status = co_registration_status or "unverified"
        self.temporal_baseline_days = temporal_baseline_days
        self.extra_metadata = extra_metadata or {}

        # Strict validation based on pair_type
        if self.pair_type == PairType.OPTICAL_SAR:
            if not (self.primary_image.is_optical() and self.secondary_image.is_sar()):
                if self.primary_image.is_sar() and self.secondary_image.is_optical():
                    # Canonical ordering: primary=optical, secondary=sar
                    self.primary_image, self.secondary_image = self.secondary_image, self.primary_image
                else:
                    raise ModalityMismatchError(
                        f"OPTICAL_SAR pair requires 1 optical and 1 SAR image, got '{self.primary_image.modality.value}' and '{self.secondary_image.modality.value}'"
                    )
        elif self.pair_type == PairType.OPTICAL_TEMPORAL:
            if not (self.primary_image.is_optical() and self.secondary_image.is_optical()):
                raise ModalityMismatchError(
                    f"OPTICAL_TEMPORAL pair requires both images to be optical, got '{self.primary_image.modality.value}' and '{self.secondary_image.modality.value}'"
                )
        elif self.pair_type == PairType.SAR_TEMPORAL:
            if not (self.primary_image.is_sar() and self.secondary_image.is_sar()):
                raise ModalityMismatchError(
                    f"SAR_TEMPORAL pair requires both images to be SAR, got '{self.primary_image.modality.value}' and '{self.secondary_image.modality.value}'"
                )

        # Calculate spatial IoU if bounds are available for both
        if spatial_overlap_iou is not None:
            self.spatial_overlap_iou = float(spatial_overlap_iou)
        elif self.primary_image.spatial_bounds and self.secondary_image.spatial_bounds:
            self.spatial_overlap_iou = self.primary_image.spatial_bounds.calculate_intersection_iou(
                self.secondary_image.spatial_bounds
            )
        else:
            self.spatial_overlap_iou = None

    @property
    def optical_image(self) -> Optional[OntologyImage]:
        if self.primary_image.is_optical():
            return self.primary_image
        if self.secondary_image.is_optical():
            return self.secondary_image
        return None

    @property
    def sar_image(self) -> Optional[OntologyImage]:
        if self.secondary_image.is_sar():
            return self.secondary_image
        if self.primary_image.is_sar():
            return self.primary_image
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_type": self.pair_type.value,
            "primary_image": self.primary_image.to_dict(),
            "secondary_image": self.secondary_image.to_dict(),
            "optical_image": self.optical_image.to_dict() if self.optical_image else None,
            "sar_image": self.sar_image.to_dict() if self.sar_image else None,
            "co_registration_status": self.co_registration_status,
            "spatial_overlap_iou": self.spatial_overlap_iou,
            "temporal_baseline_days": self.temporal_baseline_days,
            "extra_metadata": self.extra_metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent, default=str)
        except Exception as e:
            raise SchemaSerializationError(f"Failed to serialize PairedOntologyImage: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PairedOntologyImage:
        try:
            pair_type_str = data.get("pair_type", PairType.OPTICAL_SAR.value)
            pair_type = PairType.from_str(pair_type_str)

            if "primary_image" in data and "secondary_image" in data:
                primary = OntologyImage.from_dict(data["primary_image"])
                secondary = OntologyImage.from_dict(data["secondary_image"])
                return cls(
                    pair_id=str(data["pair_id"]),
                    primary_image=primary,
                    secondary_image=secondary,
                    pair_type=pair_type,
                    co_registration_status=data.get("co_registration_status", "unverified"),
                    spatial_overlap_iou=float(data["spatial_overlap_iou"]) if data.get("spatial_overlap_iou") is not None else None,
                    temporal_baseline_days=float(data["temporal_baseline_days"]) if data.get("temporal_baseline_days") is not None else None,
                    extra_metadata=data.get("extra_metadata", {}),
                )
            elif "optical_image" in data and "sar_image" in data:
                opt = OntologyImage.from_dict(data["optical_image"])
                sar = OntologyImage.from_dict(data["sar_image"])
                return cls(
                    pair_id=str(data["pair_id"]),
                    optical_image=opt,
                    sar_image=sar,
                    pair_type=PairType.OPTICAL_SAR,
                    co_registration_status=data.get("co_registration_status", "unverified"),
                    spatial_overlap_iou=float(data["spatial_overlap_iou"]) if data.get("spatial_overlap_iou") is not None else None,
                    extra_metadata=data.get("extra_metadata", {}),
                )
            else:
                raise InvalidMetadataError("paired_images", "Missing image dictionary entries in pair data")
        except Exception as e:
            if isinstance(e, (InvalidMetadataError, ModalityMismatchError)):
                raise
            raise SchemaSerializationError(f"Failed to deserialize PairedOntologyImage: {str(e)}")

    @classmethod
    def from_json(cls, json_str: str) -> PairedOntologyImage:
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise SchemaSerializationError(f"Malformed JSON string for PairedOntologyImage: {str(e)}")
