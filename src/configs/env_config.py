from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    ENV_STATE: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class GlobalConfig(BaseConfig):
    # DATABASE_URL: Optional[str] = None
    # DB_FORCE_ROLL_BACK: bool = False
    # LOGTAIL_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    ALLOWED_ISSUERS: str = ""
    ALLOWED_HOSTS: str = ""
    REDIS_URL: Optional[str] = None

    @property
    def get_allowed_issuers(self) -> list[str]:
        return (
            [issuer.strip() for issuer in self.ALLOWED_ISSUERS.split(",")]
            if self.ALLOWED_ISSUERS
            else []
        )

    @property
    def get_allowed_hosts(self) -> list[str]:
        return (
            [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
            if self.ALLOWED_HOSTS
            else []
        )


class DevConfig(GlobalConfig):
    ENV_STATE: str = "dev"
    model_config = SettingsConfigDict(env_prefix="DEV_")


class ProdConfig(GlobalConfig):
    ENV_STATE: str = "prod"
    model_config = SettingsConfigDict(env_prefix="PROD_")


class EnvTestConfig(GlobalConfig):
    # DATABASE_URL: str = "sqlite:///test.db"
    # DB_FORCE_ROLL_BACK: bool = True
    ENV_STATE: str = "test"
    model_config = SettingsConfigDict(env_prefix="TEST_")


@lru_cache()
def get_config(env_state: str):
    """Instantiate config based on the environment."""
    configs = {"dev": DevConfig, "prod": ProdConfig, "test": EnvTestConfig}
    return configs[env_state]()


config = get_config(BaseConfig().ENV_STATE)
