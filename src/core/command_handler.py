from enum import Enum


class Command(str, Enum):
    REMEMBER = "remember"
    WRITE_ABOUT_ME = "write_about_me"
    DELETE_ABOUT_ME = "delete_about_me"
    DELETE_INSTRUCTION = "delete_instruction"
    RECORD_THOUGHT = "record_thought"
    ASK_TERMINAL = "ask_terminal"
    NONE = "none"


TRIGGERS = [
    ("запомни", Command.REMEMBER),
    ("запиши обо мне", Command.WRITE_ABOUT_ME),
    ("удали обо мне", Command.DELETE_ABOUT_ME),
    ("удали инструкцию", Command.DELETE_INSTRUCTION),
    ("запиши мысль", Command.RECORD_THOUGHT),
    ("спроси терминал", Command.ASK_TERMINAL),
]


class CommandHandler:
    @staticmethod
    def detect(text: str) -> Command:
        lower = text.strip().lower()
        for trigger, command in TRIGGERS:
            if lower.startswith(trigger):
                return command
        return Command.NONE

    @staticmethod
    def extract_payload(text: str, command: Command) -> str:
        lower = text.strip().lower()
        for trigger, cmd in TRIGGERS:
            if cmd == command and lower.startswith(trigger):
                payload = text.strip()[len(trigger):].lstrip(": ")
                return payload
        return text
