"""Unit tests for ModelRegistry (`src/ml/model_registry.py`)."""

from pathlib import Path
import json
import pytest

from src.ml.model_registry import ModelMetadata, ModelValidationException, ModelRegistry


def test_model_registry_registration_and_sha256(temp_workspace: Path) -> None:
    """Verify registration, SHA256 computation, and metadata retrieval."""
    reg_dir = temp_workspace / "models"
    reg = ModelRegistry(base_dir=str(reg_dir))

    # Create dummy model file
    dummy_model = reg_dir / "rf_assignment_v1.joblib"
    with open(dummy_model, "wb") as f:
        f.write(b"MOCK_JOBLIB_BINARY_DATA_FOR_SHA256_TESTing_123456789")

    meta = reg.register_model(
        model_name="rf_classifier",
        version="v1.0.0",
        training_dataset_uri="datasets/synthetic/v1/incidents.csv",
        dataset_version="v1",
        hyperparameters={"n_estimators": 100, "max_depth": 15},
        metrics={"accuracy": 0.88, "f1_macro": 0.86},
        features_used=["priority", "category", "subcategory", "business_service"],
        target_variable="assignment_group",
        model_file_path=str(dummy_model),
        status="Active"
    )

    assert meta.model_name == "rf_classifier"
    assert len(meta.sha256_checksum) == 64
    assert reg.get_model_metadata("rf_classifier", "latest") is not None


def test_model_verification_pass_and_sha256_mismatch(temp_workspace: Path) -> None:
    """Verify SHA256 validation compliance and tamper rejection."""
    reg_dir = temp_workspace / "models"
    reg = ModelRegistry(base_dir=str(reg_dir))

    dummy_model = reg_dir / "rf_clean.joblib"
    with open(dummy_model, "wb") as f:
        f.write(b"ORIGINAL_CLEAN_MODEL_BYTES")

    reg.register_model(
        model_name="rf_clean",
        version="v1.0",
        training_dataset_uri="datasets/v1/incidents.csv",
        dataset_version="v1",
        hyperparameters={},
        metrics={"accuracy": 0.90},
        features_used=["priority", "category"],
        target_variable="assignment_group",
        model_file_path=str(dummy_model)
    )

    # Clean verification passes
    verified_path = reg.verify_and_load_model_path("rf_clean", "v1.0")
    assert verified_path.exists()

    # Tamper with file on disk
    with open(dummy_model, "ab") as f:
        f.write(b"TAMPERED_EXTRA_BYTES_INJECTED")

    with pytest.raises(ModelValidationException, match="SHA256 Checksum Mismatch"):
        reg.verify_and_load_model_path("rf_clean", "v1.0")


def test_model_verification_feature_leakage_rejection(temp_workspace: Path) -> None:
    """Verify that a model attempting to use a blocked leakage feature is rejected during verification."""
    reg_dir = temp_workspace / "models"
    reg = ModelRegistry(base_dir=str(reg_dir))

    dummy_model = reg_dir / "rf_leaky.joblib"
    with open(dummy_model, "wb") as f:
        f.write(b"LEAKY_MODEL_BYTES")

    reg.register_model(
        model_name="rf_leaky",
        version="v1.0",
        training_dataset_uri="datasets/v1/incidents.csv",
        dataset_version="v1",
        hyperparameters={},
        metrics={"accuracy": 0.99},
        features_used=["priority", "close_notes"],  # close_notes is blocked!
        target_variable="assignment_group",
        model_file_path=str(dummy_model)
    )

    with pytest.raises(ModelValidationException, match="uses blocked target leakage feature 'close_notes'"):
        reg.verify_and_load_model_path("rf_leaky", "v1.0")
