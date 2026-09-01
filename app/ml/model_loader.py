"""
Enterprise HR AI — Model Loader
Loads and caches the trained model, metadata, and feature names.
"""
import json
from pathlib import Path

import joblib

from app.utils.config import MODEL_PATH, METADATA_PATH
from app.utils.logger import model_logger


class ModelLoader:
    """Singleton model loader — avoids reloading on every request."""

    _instance = None
    _model = None
    _metadata = None
    _feature_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model pipeline and metadata from disk."""
        if self._model is not None:
            return self._model, self._metadata, self._feature_names

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run the ML training notebooks first."
            )

        model_logger.info("Loading attrition prediction model...")
        pipeline = joblib.load(str(MODEL_PATH))
        self._model = pipeline["model"]
        self._feature_names = pipeline["feature_names"]

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"version": "unknown"}

        model_logger.info(
            "Model loaded: %s v%s (%d features)",
            self._metadata.get("algorithm", "?"),
            self._metadata.get("version", "?"),
            len(self._feature_names),
        )
        return self._model, self._metadata, self._feature_names

    def get_model(self):
        self.load()
        return self._model

    def get_metadata(self):
        self.load()
        return self._metadata

    def get_feature_names(self):
        self.load()
        return self._feature_names

    def is_loaded(self):
        return self._model is not None

    def reset(self):
        """Force reload on next access."""
        self._model = None
        self._metadata = None
        self._feature_names = None
