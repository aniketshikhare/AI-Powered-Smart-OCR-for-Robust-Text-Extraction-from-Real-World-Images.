from smart_ocr.modules import preprocessing, recognition


def test_recognises_clean_text(text_image):
    pre = preprocessing.preprocess(text_image)
    result = recognition.recognise_image(pre.image)
    assert "HELLO" in result.raw_text.upper()
    assert result.mean_confidence > 0


def test_result_groups_words_into_lines():
    words = [
        recognition.RecognisedWord("HELLO", 90, (0, 0, 10, 10), line_id=0),
        recognition.RecognisedWord("WORLD", 80, (12, 0, 10, 10), line_id=0),
        recognition.RecognisedWord("NEXT", 70, (0, 20, 10, 10), line_id=1),
    ]
    result = recognition.RecognitionResult(words=words, engine="test")
    assert result.raw_text == "HELLO WORLD\nNEXT"
    assert result.mean_confidence == 80.0


def test_engine_cache_returns_same_instance():
    assert recognition.get_engine("tesseract") is recognition.get_engine("tesseract")
