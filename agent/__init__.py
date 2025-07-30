"""
Personal Work Agent

An intelligent agent that analyzes data, learns from unstructured work information,
and provides assistance based on learned knowledge.
"""

from .core.agent import PersonalWorkAgent, AgentResponse
from .utils.config import Config

__version__ = "1.0.0"
__author__ = "Personal Work Agent"
__description__ = "An AI agent that learns from your work data and provides intelligent assistance"

__all__ = [
    "PersonalWorkAgent",
    "AgentResponse", 
    "Config"
]