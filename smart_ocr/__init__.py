"""AI-Powered Smart OCR for Robust Text Extraction from Real-World Images."""
from __future__ import annotations

from flask import Flask

from .config import Config
from .database import Database
from .pipeline import OCRPipeline

__version__ = "1.0.0"


def create_app(config: type[Config] = Config) -> Flask:
    config.ensure_dirs()
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config)

    app.extensions["smart_ocr_db"] = Database(config.DATABASE_PATH)
    app.extensions["smart_ocr_pipeline"] = OCRPipeline(
        engine=config.OCR_ENGINE,
        languages=config.OCR_LANGUAGES,
        min_confidence=config.MIN_CONFIDENCE,
        debug_dir=config.DEBUG_DIR,
    )

    from .routes import bp

    app.register_blueprint(bp)
    return app
