from smart_ocr.modules import postprocessing as post
from smart_ocr.modules.recognition import RecognisedWord, RecognitionResult


def words(*items):
    return RecognitionResult(
        words=[RecognisedWord(t, c, (0, 0, 1, 1), line) for t, c, line in items], engine="test"
    )


def test_confidence_filter_drops_low_scores():
    result = post.postprocess(words(("GOOD", 95, 0), ("g@rb", 5, 0)), min_confidence=40, spell_correct=False)
    assert result.text == "GOOD"
    assert result.removed_tokens == 1


def test_character_confusion_fix():
    assert post.fix_character_confusions("H0USE") == "HOUSE"
    assert post.fix_character_confusions("10O") == "100"
    assert post.fix_character_confusions("A1") == "A1"


def test_numeric_context_fix():
    assert post.fix_numeric_context("I4PUNE") == "14PUNE"
    assert post.fix_numeric_context("1O") == "10"
    assert post.fix_numeric_context("GATE") == "GATE"
    assert post.fix_numeric_context("HOTEL9") == "HOTEL9"


def test_bundled_dictionary_is_available():
    words = post.load_dictionary(post.BUNDLED_DICTIONARY_PATH)
    assert {"platform", "invoice", "pune"} <= words


def test_whitespace_normalisation():
    assert post.normalise_whitespace("Hello   world  ,ok\n\n\n\nnext") == "Hello world, ok\nnext"


def test_spell_correction_uses_dictionary():
    vocab = {"invoice", "total", "amount"}
    assert post.correct_word("invoce", vocab) == "invoice"
    assert post.correct_word("Amount", vocab) == "Amount"
    assert post.correct_word("XZQW", vocab) == "XZQW"


def test_postprocess_reports_corrections():
    result = post.postprocess(
        words(("invoce", 90, 0), ("total", 92, 0)), spell_correct=True, dictionary={"invoice", "total"}
    )
    assert result.text == "invoice total"
    assert result.corrections == [("invoce", "invoice")]
    assert result.word_count == 2
    assert result.confidence == 91.0


def test_noise_tokens_are_dropped():
    result = post.postprocess(words(("TEXT", 90, 0), ("~", 85, 0)), spell_correct=False)
    assert result.text == "TEXT"
