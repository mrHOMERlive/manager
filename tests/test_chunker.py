import pytest
from src.utils.chunker import chunk_text


class TestChunkText:
    def test_short_text_no_split(self):
        text = "Hello, world!"
        result = chunk_text(text, max_size=3500)
        assert result == [text]

    def test_empty_text(self):
        result = chunk_text("")
        assert result == [""]

    def test_exact_max_size_no_split(self):
        text = "a" * 3500
        result = chunk_text(text, max_size=3500)
        assert result == [text]

    def test_long_text_splits(self):
        text = "a" * 7000
        result = chunk_text(text, max_size=3500)
        assert len(result) == 2
        assert "".join(result) == text

    def test_splits_at_newline(self):
        # Build text where a newline sits within the max_size window
        line1 = "a" * 2000
        line2 = "b" * 2000
        text = line1 + "\n" + line2
        result = chunk_text(text, max_size=2500)
        assert len(result) == 2
        assert result[0] == line1
        assert result[1] == line2

    def test_newline_too_early_falls_back_to_max_size(self):
        """If the only newline is before half of max_size, split at max_size instead."""
        text = "a" * 10 + "\n" + "b" * 5000
        result = chunk_text(text, max_size=100)
        # The newline at position 10 is < max_size//2 (50), so split at 100
        assert len(result[0]) == 100

    def test_multiple_chunks(self):
        text = "a" * 10000
        result = chunk_text(text, max_size=3500)
        assert len(result) == 3
        joined = "".join(result)
        assert joined == text

    def test_custom_max_size(self):
        text = "hello " * 100  # 600 chars
        result = chunk_text(text, max_size=200)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 200
