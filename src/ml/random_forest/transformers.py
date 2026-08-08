"""
Scikit-Learn Compatible Custom Transformers (`v1.5.0`).

Provides enterprise-grade preprocessing classes that implement `BaseEstimator` and `TransformerMixin`.
Enables persisting complete end-to-end `sklearn.pipeline.Pipeline` objects (preprocessing + model)
to disk (`joblib.dump`), allowing zero-manual-preprocessing inference at prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of column names from a pandas DataFrame and returns a DataFrame or numpy array.
    Guarantees consistent column ordering and safe handling of missing columns during inference.
    """

    def __init__(self, attribute_names: List[str], return_array: bool = False) -> None:
        self.attribute_names = attribute_names
        self.return_array = return_array

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "DataFrameSelector":
        """Store input feature names."""
        self.feature_names_in_ = np.array(self.attribute_names, dtype=object)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> Union[pd.DataFrame, np.ndarray]:
        """Select specified columns, filling any missing features with default neutral values."""
        if isinstance(X, np.ndarray):
            return X

        df = X.copy()
        for col in self.attribute_names:
            if col not in df.columns:
                # Fill missing numerical/categorical with neutral unknown/0
                df[col] = "UNKNOWN" if col in ["category", "subcategory", "business_service", "location", "cmdb_ci", "vendor", "contact_type"] else 0
                logger.debug(f"DataFrameSelector filled missing column '{col}' with neutral default.")

        selected = df[self.attribute_names]
        return selected.to_numpy() if self.return_array else selected

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names corresponding to selected attributes."""
        return np.array(self.attribute_names, dtype=object)


class EnterpriseFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts non-linear interactions (`priority_x_impact`, `priority_x_urgency`) and cyclic temporal shifts
    if raw columns are present. Ensures raw prediction payloads can be ingested without external engineering steps.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "EnterpriseFeatureExtractor":
        """Record input feature names and compute output feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x_{i}" for i in range(X.shape[1])], dtype=object)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> pd.DataFrame:
        """Extract interactions and cyclic shifts safely."""
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)
        df = X.copy()

        # Consolidate textual context into a single dense representation
        if "short_description" not in df.columns:
            df["short_description"] = ""
        if "description" not in df.columns:
            df["description"] = ""
            
        sd = df["short_description"].astype(str).fillna("")
        d = df["description"].astype(str).fillna("")
        df["combined_text"] = sd + " " + d
        df["combined_text"] = df["combined_text"].str.lower().str.strip()
        
        # Prevent TfidfVectorizer 'empty vocabulary' ValueError on dummy/empty datasets
        df.loc[df["combined_text"] == "", "combined_text"] = "missingtext"

        # Extract interaction terms if raw numeric columns present
        if "priority" in df.columns and "impact" in df.columns and "priority_x_impact" not in df.columns:
            df["priority_x_impact"] = pd.to_numeric(df["priority"], errors="coerce").fillna(3) * pd.to_numeric(df["impact"], errors="coerce").fillna(2)
        if "priority" in df.columns and "urgency" in df.columns and "priority_x_urgency" not in df.columns:
            df["priority_x_urgency"] = pd.to_numeric(df["priority"], errors="coerce").fillna(3) * pd.to_numeric(df["urgency"], errors="coerce").fillna(2)

        # Extract cyclic time shifts if opened_at is present and shifts are missing
        if "opened_at" in df.columns and "opened_at_hour_sin" not in df.columns:
            opened_dt = pd.to_datetime(df["opened_at"], errors="coerce").fillna(pd.Timestamp.now())
            hours = opened_dt.dt.hour
            dows = opened_dt.dt.dayofweek
            df["opened_at_hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            df["opened_at_hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            df["opened_at_dayofweek_sin"] = np.sin(2 * np.pi * dows / 7.0)
            df["opened_at_dayofweek_cos"] = np.cos(2 * np.pi * dows / 7.0)

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return exact feature names outputted after interaction and cyclic shift extraction."""
        in_feats = list(input_features) if input_features is not None else (list(self.feature_names_in_) if hasattr(self, "feature_names_in_") else [])
        out_feats = list(in_feats)
        if "priority" in in_feats and "impact" in in_feats and "priority_x_impact" not in out_feats:
            out_feats.append("priority_x_impact")
        if "priority" in in_feats and "urgency" in in_feats and "priority_x_urgency" not in out_feats:
            out_feats.append("priority_x_urgency")
        if "opened_at" in in_feats and "opened_at_hour_sin" not in out_feats:
            out_feats.extend(["opened_at_hour_sin", "opened_at_hour_cos", "opened_at_dayofweek_sin", "opened_at_dayofweek_cos"])
        return np.array(out_feats, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Learns normalized frequency distributions (`count / total_rows`) for high-cardinality categorical
    columns during training (`fit`), and maps them cleanly during inference (`transform`).
    Handles unknown inference labels by assigning `0.0001` (minimum smoothing frequency).
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns
        self.mapping_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "FrequencyEncoder":
        """Fit normalized frequency distributions across specified columns."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        cols = self.columns or X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.mapping_ = {}

        for col in cols:
            if col in X.columns:
                series = X[col].fillna("UNKNOWN").astype(str)
                counts = series.value_counts(normalize=True).to_dict()
                self.mapping_[col] = counts

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> pd.DataFrame:
        """Map categorical values to learned frequency numbers."""
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)
        df = X.copy()

        for col, mapping in self.mapping_.items():
            if col in df.columns:
                series = df[col].fillna("UNKNOWN").astype(str)
                df[col] = series.map(mapping).fillna(0.0001).astype(float)

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names (identical to input feature names as encoding is in-place)."""
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(self.columns or [], dtype=object)


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed Out-of-Fold Target Encoder for high-cardinality categorical attributes (`subcategory`, `business_service`).
    Computes `(count * mean + smoothing * global_mean) / (count + smoothing)` during `fit`.
    Handles unknown inference labels by imputing the learned `global_mean_`.
    """

    def __init__(self, columns: Optional[List[str]] = None, smoothing: float = 10.0) -> None:
        self.columns = columns
        self.smoothing = smoothing
        self.mapping_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray, List[Any]]) -> "SmoothedTargetEncoder":
        """Compute smoothed target averages across categories."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        y_series = pd.Series(y).reset_index(drop=True)
        # If classification (string/object targets like assignment_group), convert to numeric factor codes for target encoding
        if y_series.dtype == "object" or y_series.dtype == "string":
            y_series = pd.Series(pd.factorize(y_series)[0]).astype(float)
        else:
            y_series = pd.to_numeric(y_series, errors="coerce").fillna(0.0)

        self.global_mean_ = float(y_series.mean())
        cols = self.columns or X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.mapping_ = {}

        for col in cols:
            if col in X.columns:
                series = X[col].reset_index(drop=True).fillna("UNKNOWN").astype(str)
                grouped = pd.DataFrame({"cat": series, "target": y_series}).groupby("cat")["target"]
                counts = grouped.count()
                means = grouped.mean()
                smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
                self.mapping_[col] = smoothed.to_dict()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> pd.DataFrame:
        """Map categories to learned smoothed target values."""
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)
        df = X.copy()

        for col, mapping in self.mapping_.items():
            if col in df.columns:
                series = df[col].fillna("UNKNOWN").astype(str)
                df[col] = series.map(mapping).fillna(self.global_mean_).astype(float)

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names (identical to input feature names as encoding is in-place)."""
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(self.columns or [], dtype=object)
