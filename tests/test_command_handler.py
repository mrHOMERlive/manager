import pytest
from src.core.command_handler import CommandHandler, Command


class TestCommandDetect:
    def test_detect_remember(self):
        assert CommandHandler.detect("запомни что я люблю кофе") == Command.REMEMBER

    def test_detect_write_about_me(self):
        assert CommandHandler.detect("запиши обо мне: я программист") == Command.WRITE_ABOUT_ME

    def test_detect_delete_about_me(self):
        assert CommandHandler.detect("удали обо мне запись 1") == Command.DELETE_ABOUT_ME

    def test_detect_delete_instruction(self):
        assert CommandHandler.detect("удали инструкцию 3") == Command.DELETE_INSTRUCTION

    def test_detect_record_thought(self):
        assert CommandHandler.detect("запиши мысль: нужно купить молоко") == Command.RECORD_THOUGHT

    def test_detect_ask_terminal(self):
        assert CommandHandler.detect("спроси терминал какая погода") == Command.ASK_TERMINAL

    def test_detect_none_for_unknown(self):
        assert CommandHandler.detect("привет как дела") == Command.NONE

    def test_detect_case_insensitive(self):
        assert CommandHandler.detect("Запомни это важно") == Command.REMEMBER
        assert CommandHandler.detect("ЗАПОМНИ это важно") == Command.REMEMBER

    def test_detect_with_leading_whitespace(self):
        assert CommandHandler.detect("  запомни пробелы") == Command.REMEMBER


class TestExtractPayload:
    def test_extract_remember_payload(self):
        text = "запомни что я люблю кофе"
        payload = CommandHandler.extract_payload(text, Command.REMEMBER)
        assert payload == "что я люблю кофе"

    def test_extract_payload_with_colon_separator(self):
        text = "запиши обо мне: я программист"
        payload = CommandHandler.extract_payload(text, Command.WRITE_ABOUT_ME)
        assert payload == "я программист"

    def test_extract_payload_no_match_returns_full_text(self):
        text = "обычный текст"
        payload = CommandHandler.extract_payload(text, Command.REMEMBER)
        assert payload == "обычный текст"

    def test_extract_record_thought_payload(self):
        text = "запиши мысль: нужно купить молоко"
        payload = CommandHandler.extract_payload(text, Command.RECORD_THOUGHT)
        assert payload == "нужно купить молоко"

    def test_extract_ask_terminal_payload(self):
        text = "спроси терминал какая погода"
        payload = CommandHandler.extract_payload(text, Command.ASK_TERMINAL)
        assert payload == "какая погода"

    def test_extract_payload_preserves_original_case(self):
        text = "Запомни Важную Информацию"
        payload = CommandHandler.extract_payload(text, Command.REMEMBER)
        assert payload == "Важную Информацию"
