"""
Unit tests for SatQuery AI preprocessing pipeline.

Tests:
1. Sentinel-2 multi-resolution band handling
2. Sentinel-1 SAR preprocessing
3. GeoTIFF validation
4. VLM RGB output format
"""

import numpy as np
import pytest


class TestSentinel2Preprocessing:
    """Tests for the multi-resolution Sentinel-2 preprocessor."""

    def _make_bands_dict(self):
        """Create a mock BigEarthNet v2 band dictionary with correct dimensions."""
        rng = np.random.RandomState(42)
        return {
            # 10m bands: 120x120
            "B02": rng.uniform(100, 3000, (120, 120)).astype(np.float32),
            "B03": rng.uniform(100, 3000, (120, 120)).astype(np.float32),
            "B04": rng.uniform(100, 3000, (120, 120)).astype(np.float32),
            "B08": rng.uniform(100, 3000, (120, 120)).astype(np.float32),
            # 20m bands: 60x60
            "B05": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            "B06": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            "B07": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            "B8A": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            "B11": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            "B12": rng.uniform(200, 4000, (60, 60)).astype(np.float32),
            # 60m bands: 20x20
            "B01": rng.uniform(100, 1000, (20, 20)).astype(np.float32),
            "B09": rng.uniform(100, 5000, (20, 20)).astype(np.float32),
        }

    def test_harmonization_output_shape(self):
        """All bands should be harmonized to 10m resolution (120x120)."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor, S2PreprocessConfig,
        )
        config = S2PreprocessConfig(use_60m_bands=False)
        preprocessor = Sentinel2Preprocessor(config)
        bands = self._make_bands_dict()

        result = preprocessor.process_bands(bands)

        # Should have 10 bands (4 at 10m + 6 at 20m, no 60m)
        assert result["spectral_harmonized"].shape[0] == 10
        # All should be 120x120
        assert result["spectral_harmonized"].shape[1] == 120
        assert result["spectral_harmonized"].shape[2] == 120

    def test_vlm_rgb_output_format(self):
        """VLM RGB should be (H, W, 3) uint8 [0, 255]."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor, S2PreprocessConfig,
        )
        preprocessor = Sentinel2Preprocessor()
        bands = self._make_bands_dict()

        result = preprocessor.process_bands(bands)
        vlm_rgb = result["vlm_rgb"]

        assert vlm_rgb.shape == (120, 120, 3)
        assert vlm_rgb.dtype == np.uint8
        assert vlm_rgb.min() >= 0
        assert vlm_rgb.max() <= 255

    def test_60m_bands_dropped_by_default(self):
        """60m bands (B01, B09) should be dropped by default."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor,
        )
        preprocessor = Sentinel2Preprocessor()  # Default config
        bands = self._make_bands_dict()

        result = preprocessor.process_bands(bands)
        provenance = result["provenance"]

        assert "B01" in provenance.bands_dropped
        assert "B09" in provenance.bands_dropped
        assert "B01" not in provenance.bands_used
        assert "B09" not in provenance.bands_used

    def test_ndvi_computed(self):
        """NDVI should be computed from B08 (NIR) and B04 (Red)."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor, S2PreprocessConfig,
        )
        config = S2PreprocessConfig(compute_ndvi=True)
        preprocessor = Sentinel2Preprocessor(config)
        bands = self._make_bands_dict()

        result = preprocessor.process_bands(bands)

        assert "NDVI" in result["spectral_indices"]
        ndvi = result["spectral_indices"]["NDVI"]
        assert ndvi.shape == (120, 120)
        # NDVI should be in [-1, 1]
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_20m_bands_upsampled(self):
        """20m bands (60x60) should be upsampled to 120x120."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor, S2PreprocessConfig,
        )
        config = S2PreprocessConfig()
        preprocessor = Sentinel2Preprocessor(config)

        # Must include the VLM RGB bands (B04, B03, B02) plus a 20m band
        bands = {
            "B02": np.ones((120, 120), dtype=np.float32) * 400,
            "B03": np.ones((120, 120), dtype=np.float32) * 600,
            "B04": np.ones((120, 120), dtype=np.float32) * 500,
            "B11": np.ones((60, 60), dtype=np.float32) * 1000,
        }

        result = preprocessor.process_bands(bands)
        # Should have 4 bands (B02, B03, B04 at 10m + B11 upsampled) all at 120x120
        assert result["spectral_harmonized"].shape == (4, 120, 120)

    def test_provenance_logged(self):
        """Preprocessing provenance should be complete."""
        from services.models.preprocessing.sentinel2_preprocessing import (
            Sentinel2Preprocessor,
        )
        preprocessor = Sentinel2Preprocessor()
        bands = self._make_bands_dict()

        result = preprocessor.process_bands(bands)
        prov = result["provenance"]

        assert len(prov.bands_used) > 0
        assert prov.resample_method == "bilinear"
        assert prov.target_resolution_m == 10


class TestSentinel1Preprocessing:
    """Tests for the SAR preprocessor."""

    def _make_sar_channels(self):
        rng = np.random.RandomState(42)
        return {
            "VV": rng.uniform(-25, 5, (120, 120)).astype(np.float32),
            "VH": rng.uniform(-35, -5, (120, 120)).astype(np.float32),
        }

    def test_pseudo_rgb_output(self):
        """SAR pseudo-RGB should be (H, W, 3) uint8."""
        from services.models.preprocessing.sentinel1_preprocessing import (
            Sentinel1Preprocessor,
        )
        preprocessor = Sentinel1Preprocessor()
        channels = self._make_sar_channels()

        result = preprocessor.process_channels(channels)
        rgb = result["vlm_rgb"]

        assert rgb.shape == (120, 120, 3)
        assert rgb.dtype == np.uint8

    def test_normalization(self):
        """Normalized SAR should have reasonable values (near 0 mean)."""
        from services.models.preprocessing.sentinel1_preprocessing import (
            Sentinel1Preprocessor,
        )
        preprocessor = Sentinel1Preprocessor()
        channels = self._make_sar_channels()

        result = preprocessor.process_channels(channels)
        sar = result["sar_normalized"]

        assert sar.shape[0] == 2  # VH, VV (sorted alphabetically)
        # After z-score normalization, mean should be near 0
        assert abs(np.mean(sar[0])) < 2.0
        assert abs(np.mean(sar[1])) < 2.0


class TestGeoTIFFValidator:
    """Tests for GeoTIFF validation."""

    def test_format_compliance_png_rejected(self, tmp_path):
        """PNG should be rejected unless from a benchmark dataset."""
        from PIL import Image
        from services.models.preprocessing.geotiff_utils import GeoTIFFValidator

        # Create a test PNG
        img = Image.new("RGB", (100, 100), color="red")
        png_path = tmp_path / "test.png"
        img.save(str(png_path))

        validator = GeoTIFFValidator()
        meta = validator.validate_and_extract(str(png_path))

        # Should have validation error (no dataset_source)
        assert len(meta.validation_errors) > 0
        assert "benchmark" in meta.validation_errors[0].lower()

    def test_format_compliance_png_accepted_for_benchmark(self, tmp_path):
        """PNG should be accepted when source is a benchmark dataset."""
        from PIL import Image
        from services.models.preprocessing.geotiff_utils import GeoTIFFValidator

        img = Image.new("RGB", (100, 100), color="blue")
        png_path = tmp_path / "benchmark.png"
        img.save(str(png_path))

        validator = GeoTIFFValidator()
        meta = validator.validate_and_extract(str(png_path), dataset_source="VRSBench")

        assert len(meta.validation_errors) == 0
        assert meta.is_benchmark_format is True

    def test_modality_inference_sar(self, tmp_path):
        """2-band image should be inferred as SAR."""
        from services.models.preprocessing.geotiff_utils import (
            GeoTIFFValidator, ImageModality,
        )
        from PIL import Image

        # Create a 2-band test image
        img = Image.new("LA", (120, 120))  # 2-channel: L + Alpha
        tif_path = tmp_path / "sar_test.tif"
        img.save(str(tif_path))

        validator = GeoTIFFValidator()
        meta = validator.validate_and_extract(str(tif_path))

        assert meta.modality == ImageModality.SAR


class TestContractSchemas:
    """Test that response models match contract schemas."""

    def test_vqa_response_schema(self):
        """VQA response should match the contract schema."""
        from services.models.api.app import VQAResponse, Confidence, Provenance

        response = VQAResponse(
            answer="Yes, there is a river visible.",
            confidence=Confidence(score=0.85, method="model_logit"),
            provenance=Provenance(model_name="InternVL2-2B"),
        )

        data = response.model_dump()
        assert "answer" in data
        assert "confidence" in data
        assert data["confidence"]["score"] == 0.85
        assert "provenance" in data

    def test_fusion_response_has_modality_attribution(self):
        """Fusion response MUST include modality_attribution (§6.4)."""
        from services.models.api.app import (
            FusionResponse, Confidence, Provenance, ModalityAttribution,
        )

        response = FusionResponse(
            fused_answer="Urban area with high moisture content",
            modality_attribution=ModalityAttribution(
                optical_evidence="Dense urban structures visible",
                sar_evidence="High backscatter indicating buildings",
                fused_reasoning="Both confirm built-up area",
            ),
            confidence=Confidence(score=0.75, method="model_logit"),
            provenance=Provenance(model_name="InternVL2-2B"),
        )

        data = response.model_dump()
        assert "modality_attribution" in data
        assert data["modality_attribution"]["optical_evidence"] != ""
        assert data["modality_attribution"]["sar_evidence"] != ""
        assert "disagreement_flag" in data

    def test_change_response_schema(self):
        """Change response should have required fields."""
        from services.models.api.app import ChangeResponse, Confidence, Provenance

        response = ChangeResponse(
            description="New construction detected in the southern region",
            change_detected=True,
            confidence=Confidence(score=0.72, method="model_logit"),
            provenance=Provenance(model_name="InternVL2-2B"),
        )

        data = response.model_dump()
        assert data["change_detected"] is True
        assert "description" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
