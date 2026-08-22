from smart_ocr.modules.evaluation import evaluate, normalize_text


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Hello\n  WORLD  ") == "hello world"


def test_evaluate_exact_match():
    result = evaluate("SHOP NO 14 PUNE", "shop   no 14 pune")
    assert result.exact_match is True
    assert result.cer == 0
    assert result.wer == 0
    assert result.character_accuracy == 100
    assert result.word_accuracy == 100


def test_evaluate_character_error():
    result = evaluate("PLATFORM NO 5", "PLTFORM NO 5")
    assert result.character_errors == 1
    assert result.cer > 0
    assert result.character_accuracy < 100


def test_evaluate_word_error():
    result = evaluate("SHOP NO 14 PUNE", "SHOP NO 14")
    assert result.word_errors == 1
    assert result.wer > 0
    assert result.word_accuracy < 100


def test_evaluate_empty_reference():
    result = evaluate("", "unexpected")
    assert result.exact_match is False
    assert result.cer == 100
    assert result.wer == 100
