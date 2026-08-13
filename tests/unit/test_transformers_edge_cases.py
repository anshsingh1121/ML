"""Unit tests for edge cases, numpy input fallbacks, and feature names across transformers.py (`v1.5.0`)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.catboost.transformers import (
    DataFrameSelector,
    EnterpriseFeatureExtractor,
    FrequencyEncoder,
    SmoothedTargetEncoder,
)


def test_dataframe_selector_edge_cases() -> None:
    """Test DataFrameSelector array inputs, missing columns, and feature names."""
    selector = DataFrameSelector(attribute_names=["colA", "colB"], return_array=True)
    # Numpy input bypass
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(selector.transform(arr), arr)

    # Missing column filling with defaults
    df = pd.DataFrame({"colA": [10, 20]})
    res = selector.transform(df)
    assert res.shape == (2, 2)
    assert selector.get_feature_names_out().tolist() == ["colA", "colB"]


def test_enterprise_feature_extractor_edge_cases() -> None:
    """Test EnterpriseFeatureExtractor numpy fallback and get_feature_names_out."""
    extractor = EnterpriseFeatureExtractor()
    arr = np.array([[1, 2], [3, 4]])
    out_df = extractor.fit_transform(arr)
    # Ensure original data is perfectly preserved in the output
    assert np.array_equal(out_df[["x_0", "x_1"]].values, arr)
    assert "combined_text" in out_df.columns

    df = pd.DataFrame({
        "priority": [1, 2],
        "business_impact": [2, 3],
        "urgency": [1, 2],
        "opened_at": ["2025-01-01 10:00:00", "2025-01-02 12:00:00"]
    })
    extractor.fit(df)
    out_names = extractor.get_feature_names_out(["priority", "business_impact", "urgency", "opened_at"])
    assert "priority_x_business_impact" in out_names
    assert "priority_x_urgency" in out_names
    assert "opened_at_hour_sin" in out_names


def test_frequency_encoder_edge_cases() -> None:
    """Test FrequencyEncoder with array inputs and get_feature_names_out options."""
    encoder = FrequencyEncoder(columns=["catA"])
    arr = np.array([["X"], ["Y"]])
    encoder.fit(arr)
    assert encoder.get_feature_names_out().tolist() == [0]

    df = pd.DataFrame({"catA": ["X", "X", "Y"]})
    encoder.fit(df)
    assert encoder.get_feature_names_out(input_features=["catA"]).tolist() == ["catA"]
    assert encoder.get_feature_names_out().tolist() == ["catA"]

    # Transform numpy array using learned feature_names_in_
    arr_in = np.array([["X"], ["Z"]])
    transformed = encoder.transform(arr_in)
    assert "catA" in transformed.columns
    assert transformed["catA"].iloc[1] == 0.0001


def test_smoothed_target_encoder_edge_cases() -> None:
    """Test SmoothedTargetEncoder with array inputs and get_feature_names_out options."""
    encoder = SmoothedTargetEncoder(columns=["catA"], smoothing=10.0)
    arr = np.array([["A"], ["B"]])
    y = np.array([1, 0])
    encoder.fit(arr, y)
    assert encoder.get_feature_names_out().tolist() == [0]

    df = pd.DataFrame({"catA": ["A", "B", "A"]})
    encoder.fit(df, pd.Series([1, 0, 1]))
    assert encoder.get_feature_names_out().tolist() == ["catA"]

    # Transform numpy array input
    res = encoder.transform(np.array([["A"], ["C"]]))
    assert "catA" in res.columns
