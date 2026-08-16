import cv2
import pytest

from smart_ocr.modules.upload import UploadError, is_allowed, load_image, save_upload

ALLOWED = {"png", "jpg", "jpeg"}


def encode(image):
    return cv2.imencode(".png", image)[1].tobytes()


def test_is_allowed():
    assert is_allowed("photo.PNG", ALLOWED)
    assert not is_allowed("notes.pdf", ALLOWED)
    assert not is_allowed("noextension", ALLOWED)


def test_save_upload_persists_image(tmp_path, text_image):
    result = save_upload(encode(text_image), "scene.png", tmp_path, ALLOWED, 10_000_000)
    assert result.path.exists()
    assert (result.width, result.height) == (text_image.shape[1], text_image.shape[0])
    assert result.original_name == "scene.png"
    assert load_image(result.path) is not None


@pytest.mark.parametrize(
    "data,name,max_bytes,message",
    [
        (b"x", "doc.pdf", 1000, "Unsupported"),
        (b"", "doc.png", 1000, "empty"),
        (b"not-an-image-at-all", "doc.png", 1000, "not a readable image"),
    ],
)
def test_save_upload_rejects_bad_input(tmp_path, data, name, max_bytes, message):
    with pytest.raises(UploadError) as exc:
        save_upload(data, name, tmp_path, ALLOWED, max_bytes)
    assert message.lower() in str(exc.value).lower()


def test_save_upload_rejects_large_file(tmp_path, text_image):
    with pytest.raises(UploadError, match="too large"):
        save_upload(encode(text_image), "big.png", tmp_path, ALLOWED, 10)
