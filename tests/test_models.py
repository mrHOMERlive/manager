from src.memory.models import Instruction, AboutMe, Dialogue, Thought, Download, Transcription


def test_instruction_model_fields():
    inst = Instruction(content="test rule", meta={"key": "val"})
    assert inst.content == "test rule"
    assert inst.meta == {"key": "val"}


def test_transcription_model_fields():
    t = Transcription(raw_text="hello world", source_type="voice")
    assert t.raw_text == "hello world"
    assert t.source_type == "voice"
