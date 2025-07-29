#!/usr/bin/env python3
"""
Personal Work Agent - Main Entry Point

Run this file to start the agent server with web interface and API endpoints.
"""

import uvicorn
import asyncio
import logging
from pathlib import Path

from agent.utils.config import Config
from agent.api.main import app


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point"""
    print("🤖 Starting Personal Work Agent...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config()
    
    # Validate configuration
    is_valid, errors = config.validate_config()
    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        print("❌ Please fix configuration errors before starting the agent.")
        return
    
    # Print startup information
    print(f"📋 Agent Name: {config.agent_name}")
    print(f"🌐 Server: http://{config.api_host}:{config.api_port}")
    print(f"📚 Documentation: http://{config.api_host}:{config.api_port}/docs")
    print(f"💾 Data Directory: {config.data_directory}")
    print(f"🔍 Embedding Model: {config.embedding_model}")
    
    if not config.openai_api_key:
        print("⚠️  Warning: No OpenAI API key provided. Some features may not work.")
    
    print("\n🚀 Starting server...")
    
    try:
        # Start the server
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            reload=config.debug,
            log_level=config.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Personal Work Agent...")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        print(f"❌ Failed to start server: {str(e)}")


if __name__ == "__main__":
    main()