from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    controller_name: str = "satquery-b1-controller"
    host: str = "0.0.0.0"
    port: int = 8000
    b2_base_url: str = "http://b2:8100"
    ml_base_url: str = "http://mock-ml:8200"
    confidence_base_url: str = "http://b2:8100"
    request_timeout_seconds: float = 60.0
    max_query_length: int = 2000
    model_config = SettingsConfigDict(env_prefix="SATQUERY_", env_file=".env", extra="ignore")


settings = Settings()
