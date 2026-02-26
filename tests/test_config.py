from src.core.config import Settings, load_yaml_config


def test_settings_defaults():
    s = Settings(database_url="postgresql+asyncpg://test:test@localhost/test")
    assert "localhost" in s.database_url


def test_allowed_user_ids_parsing():
    s = Settings(allowed_user_ids="123,456,789")
    assert s.allowed_user_id_list == [123, 456, 789]


def test_allowed_user_ids_empty():
    s = Settings(allowed_user_ids="")
    assert s.allowed_user_id_list == []
