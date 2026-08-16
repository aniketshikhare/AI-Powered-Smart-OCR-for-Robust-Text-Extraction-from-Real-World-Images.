import cv2
import io

from conftest import make_text_image
from smart_ocr.database import Database
from smart_ocr.modules.result import OCRResult
from smart_ocr.pipeline import OCRPipeline


def test_pipeline_end_to_end(tmp_path, text_image):
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), make_text_image("INVOICE TOTAL 250", noise=True))
    result = OCRPipeline(debug_dir=tmp_path / "debug").run(path)
    assert "INVOICE" in result.text.upper()
    assert result.confidence > 0
    assert result.elapsed_ms >= 0
    assert "denoise" in result.preprocess_steps


def test_database_crud(tmp_path):
    db = Database(tmp_path / "t.db")
    ocr_id = db.save_record("a.png", "hello", 88.5, user_id="u1")
    assert db.get_record(ocr_id)["extracted_text"] == "hello"
    assert len(db.list_records(user_id="u1")) == 1
    assert db.list_records(user_id="other") == []
    assert db.delete_record(ocr_id) is True
    assert db.get_record(ocr_id) is None


def test_result_exports():
    r = OCRResult("a.png", "s.png", "hello", 91.0, 1, 1, "tesseract")
    assert "hello" in r.to_txt()
    assert '"confidence": 91.0' in r.to_json()
    assert r.quality_label == "High"


def test_health_and_pages(client):
    assert client.get("/health").get_json() == {"status": "ok"}
    assert b"Smart OCR" in client.get("/").data
    assert client.get("/history").status_code == 200


def test_api_ocr_roundtrip(client):
    png = cv2.imencode(".png", make_text_image("RECEIPT 199"))[1].tobytes()
    res = client.post(
        "/api/ocr",
        data={"image": (io.BytesIO(png), "receipt.png")},
        content_type="multipart/form-data",
    )
    payload = res.get_json()
    assert res.status_code == 200
    assert "RECEIPT" in payload["text"].upper()
    assert payload["ocr_id"] >= 1

    txt = client.get(f"/download/{payload['ocr_id']}.txt")
    assert txt.status_code == 200
    assert b"RECEIPT" in txt.data.upper()
    assert len(client.get("/api/records").get_json()) == 1


def test_api_rejects_bad_upload(client):
    res = client.post(
        "/api/ocr",
        data={"image": (io.BytesIO(b"nope"), "x.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()
