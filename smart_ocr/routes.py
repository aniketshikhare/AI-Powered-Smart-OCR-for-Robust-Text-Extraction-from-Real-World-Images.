"""HTTP layer: upload endpoint, result view, history and exports."""
from __future__ import annotations

import io

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)

from .modules.evaluation import evaluate
from .modules.preprocessing import PreprocessOptions
from .modules.upload import UploadError, save_upload

bp = Blueprint("main", __name__)


def _bool_arg(form, name: str, default: bool = True) -> bool:
    value = form.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "on", "yes"}


@bp.get("/")
def index():
    db = current_app.extensions["smart_ocr_db"]
    return render_template("index.html", history=db.list_records(limit=10))


@bp.get("/history")
def history():
    db = current_app.extensions["smart_ocr_db"]
    return render_template("history.html", records=db.list_records(limit=100))


@bp.post("/api/ocr")
def api_ocr():
    config = current_app.config
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image field in request."}), 400

    try:
        uploaded = save_upload(
            data=file.read(),
            filename=file.filename or "",
            upload_dir=config["UPLOAD_DIR"],
            allowed=config["ALLOWED_EXTENSIONS"],
            max_bytes=config["MAX_CONTENT_LENGTH"],
        )
    except UploadError as exc:
        return jsonify({"error": str(exc)}), 400

    options = PreprocessOptions(
        denoise=_bool_arg(request.form, "denoise"),
        enhance_contrast=_bool_arg(request.form, "contrast"),
        sharpen=_bool_arg(request.form, "sharpen"),
        deskew=_bool_arg(request.form, "deskew"),
        threshold=_bool_arg(request.form, "threshold"),
        perspective_correct=_bool_arg(request.form, "perspective", default=False),
    )

    pipeline = current_app.extensions["smart_ocr_pipeline"]
    try:
        result = pipeline.run(
            uploaded.path,
            image_name=uploaded.original_name,
            options=options,
            detector=request.form.get("detector", "morphology"),
            spell_correct=_bool_arg(request.form, "spell_correct"),
            use_detection=_bool_arg(request.form, "detection"),
            engine=request.form.get("engine") or None,
        )
    except ImportError as exc:
        return jsonify({"error": f"Recognition engine dependency is missing: {exc}"}), 503
    except OSError:
        # pytesseract raises TesseractNotFoundError (an OSError subclass) when
        # the Windows Tesseract executable is missing or not on PATH.
        return jsonify({
            "error": "Tesseract OCR is not installed or is not available on PATH. "
                     "Install Tesseract OCR and restart the terminal."
        }), 503

    db = current_app.extensions["smart_ocr_db"]
    result.ocr_id = db.save_record(
        image_name=result.image_name,
        extracted_text=result.text,
        confidence_score=result.confidence,
        user_id=request.form.get("user_id", "guest"),
    )

    payload = result.to_dict()
    payload["quality_label"] = result.quality_label

    # Optional ground truth lets a project demonstrator prove extraction
    # accuracy instead of treating OCR confidence as correctness.
    ground_truth = request.form.get("ground_truth", "").strip()
    if ground_truth:
        payload["evaluation"] = evaluate(ground_truth, result.text).to_dict()

    return jsonify(payload)


@bp.post("/api/evaluate")
def api_evaluate():
    """Compare supplied OCR text against known ground truth."""
    data = request.get_json(silent=True) or request.form
    expected = str(data.get("ground_truth", "")).strip()
    predicted = str(data.get("predicted_text", ""))
    if not expected:
        return jsonify({"error": "ground_truth is required."}), 400
    return jsonify(evaluate(expected, predicted).to_dict())


@bp.get("/api/records")
def api_records():
    db = current_app.extensions["smart_ocr_db"]
    return jsonify(db.list_records(user_id=request.args.get("user_id"), limit=100))


@bp.delete("/api/records/<int:ocr_id>")
def api_delete_record(ocr_id: int):
    db = current_app.extensions["smart_ocr_db"]
    if not db.delete_record(ocr_id):
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"deleted": ocr_id})


@bp.get("/download/<int:ocr_id>.<fmt>")
def download(ocr_id: int, fmt: str):
    db = current_app.extensions["smart_ocr_db"]
    record = db.get_record(ocr_id)
    if record is None:
        return jsonify({"error": "Record not found."}), 404
    if fmt == "json":
        import json

        body = json.dumps(record, indent=2, ensure_ascii=False)
        mimetype = "application/json"
    else:
        body = record["extracted_text"]
        mimetype = "text/plain"
    return send_file(
        io.BytesIO(body.encode("utf-8")),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"ocr_{ocr_id}.{ 'json' if fmt == 'json' else 'txt' }",
    )


@bp.get("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(current_app.config["UPLOAD_DIR"], filename)


@bp.get("/preview/<path:filename>")
def preview_file(filename: str):
    return send_from_directory(current_app.config["DEBUG_DIR"], filename)


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})
