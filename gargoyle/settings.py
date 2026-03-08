"""
Configuration management module for the Gargoyle application.

This module defines the settings structure using Pydantic's BaseSettings,
allowing configuration via environment variables with a 'GARGOYLE_' prefix.
"""

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMModelSettings(BaseModel):

    """Configuration settings for an individual LLM model."""

    model: str = Field(description="The name of the model to use (e.g., 'gpt-4o').")
    api_key: SecretStr = Field(description="API key for accessing the model.")
    base_url: str | None = Field(default=None, description="Optional base URL for OpenAI-compatible APIs.")


class GraphNodesModelSettings(BaseModel):

    """Configuration for model IDs used by different graph nodes."""

    merge_keyword_hierarchies_id: str = Field(
        default="default",
        description="Model ID for merging keyword hierarchies.",
    )
    keywords_single_step_builder_id: str = Field(
        default="default",
        description="Model ID for the single-step keyword builder.",
    )
    keywords_hierarchy_builder_id: str = Field(
        default="default",
        description="Model ID for building keyword hierarchies.",
    )
    keywords_extractor_id: str = Field(
        default="default",
        description="Model ID for extracting keywords.",
    )


class RestApiSettings(BaseModel):

    """Configuration settings for the REST API."""

    host: str = Field(
        default="127.0.0.1",
        description="Host address for the REST API server.",
    )
    port: int = Field(
        default=5000,
        description="Port number for the REST API server.",
        ge=1,
        le=65535,
    )


class Settings(BaseSettings):

    """
    Main application settings class.

    Includes a mapping of configured LLM models and model identifiers for specific graph nodes.
    """

    llm_models: dict[str, LLMModelSettings] = Field(
        default_factory=dict,
        description="Mapping of model IDs to their configurations.",
    )

    graph_nodes: GraphNodesModelSettings = Field(
        default_factory=GraphNodesModelSettings,
        description="Model identifiers for different graph nodes.",
    )

    rest_api: RestApiSettings = Field(
        default_factory=RestApiSettings,
        description="Configuration settings for the REST API.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_prefix="GARGOYLE_",
    )


settings = Settings()
