import os
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Configuration class for the Personal Work Agent"""
    
    # API Keys and External Services
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./agent_data.db")
    
    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Agent Configuration
    agent_name: str = os.getenv("AGENT_NAME", "WorkAssistant")
    max_memory_size: int = int(os.getenv("MAX_MEMORY_SIZE", "1000"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Chunking Configuration
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Vector Database Configuration
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "agent.log")
    
    # File Storage
    data_directory: str = os.getenv("DATA_DIRECTORY", "./data")
    upload_directory: str = os.getenv("UPLOAD_DIRECTORY", "./data/uploads")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.data_directory,
            self.upload_directory,
            self.vector_db_path,
            Path(self.database_url.replace("sqlite:///", "")).parent if "sqlite" in self.database_url else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate configuration and return status with any errors"""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OpenAI API key is required but not provided")
        
        if not self.agent_name:
            errors.append("Agent name cannot be empty")
        
        if self.max_chunk_tokens <= 0:
            errors.append("Max chunk tokens must be positive")
        
        if self.api_port <= 0 or self.api_port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_file(cls, config_file: str) -> "Config":
        """Load configuration from a file"""
        # This could be extended to support JSON/YAML config files
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive data"""
        config_dict = self.dict()
        # Remove sensitive information
        sensitive_keys = ["openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "***HIDDEN***"
        return config_dict