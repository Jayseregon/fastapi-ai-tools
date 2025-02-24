import pytest

from src.configs.env_config import (
    BaseConfig,
    DevConfig,
    EnvTestConfig,
    GlobalConfig,
    ProdConfig,
    get_config,
)


def test_base_config(monkeypatch):
    """Test BaseConfig without environment"""
    # Clear any existing ENV_STATE from os.environ
    monkeypatch.delenv("ENV_STATE", raising=False)

    # Override .env loading by passing _env_file=""
    config = BaseConfig(_env_file="")
    assert hasattr(config, "ENV_STATE")
    assert config.ENV_STATE is None


def test_global_config_defaults():
    config = GlobalConfig()
    assert config.OPENAI_API_KEY is None
    assert config.ALLOWED_HOSTS == ""
    assert config.REDIS_URL is None


def test_global_config_allowed_hosts():
    config = GlobalConfig(ALLOWED_HOSTS="localhost,example.com")
    assert config.get_allowed_hosts == ["localhost", "example.com"]

    config = GlobalConfig(ALLOWED_HOSTS="")
    assert config.get_allowed_hosts == []


def test_environment_specific_configs():
    dev_config = DevConfig()
    prod_config = ProdConfig()
    test_config = EnvTestConfig()

    assert dev_config.model_config["env_prefix"] == "DEV_"
    assert prod_config.model_config["env_prefix"] == "PROD_"
    assert test_config.model_config["env_prefix"] == "TEST_"


@pytest.mark.parametrize(
    "env_state,expected_config",
    [
        ("dev", DevConfig),
        ("prod", ProdConfig),
        ("test", EnvTestConfig),
    ],
)
def test_get_config(env_state, expected_config, monkeypatch):
    # Clear the lru_cache
    get_config.cache_clear()
    # Set the environment state
    monkeypatch.setenv("ENV_STATE", env_state)

    config = get_config(env_state)
    assert isinstance(config, expected_config)
    assert config.ENV_STATE == env_state


def test_get_config_invalid_env(monkeypatch):
    # Clear the lru_cache
    get_config.cache_clear()
    # Set an invalid environment state
    monkeypatch.setenv("ENV_STATE", "invalid")

    # Should default to ProdConfig when invalid
    config = get_config("invalid")
    assert isinstance(config, ProdConfig)


@pytest.mark.parametrize(
    "env_vars,expected_values",
    [
        (
            {
                "TEST_OPENAI_API_KEY": "test-key",
                "TEST_ALLOWED_HOSTS": "localhost,test.com",
                "TEST_REDIS_URL": "redis://localhost",
            },
            {
                "OPENAI_API_KEY": "test-key",
                "ALLOWED_HOSTS": "localhost,test.com",
                "REDIS_URL": "redis://localhost",
            },
        ),
    ],
)
def test_environment_variables(env_vars, expected_values, monkeypatch):
    # Clear the lru_cache for get_config
    get_config.cache_clear()

    # Set test environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Create fresh config instance
    config = EnvTestConfig()
    for key, value in expected_values.items():
        assert getattr(config, key) == value
