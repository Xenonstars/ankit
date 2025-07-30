import pandas as pd
import numpy as np
import json
import logging
from typing import Any, Dict, List, Union
import re
from datetime import datetime
import mimetypes
from pathlib import Path

import nltk
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

from ..utils.config import Config


class DataAnalyzer:
    """
    Comprehensive data analyzer that can handle multiple data types
    and extract meaningful insights for the agent
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
    
    async def analyze(self, data: Any, data_type: str = "auto") -> Dict[str, Any]:
        """
        Main analysis method that routes data to appropriate analyzer
        """
        if data_type == "auto":
            data_type = self._detect_data_type(data)
        
        self.logger.info(f"Analyzing data as type: {data_type}")
        
        analysis_methods = {
            "text": self._analyze_text,
            "csv": self._analyze_csv,
            "json": self._analyze_json,
            "numeric": self._analyze_numeric,
            "file": self._analyze_file,
            "image": self._analyze_image,
            "url": self._analyze_url
        }
        
        if data_type in analysis_methods:
            return await analysis_methods[data_type](data)
        else:
            return {
                "status": "error",
                "message": f"Unsupported data type: {data_type}",
                "data_type": data_type
            }
    
    def _detect_data_type(self, data: Any) -> str:
        """Automatically detect the type of input data"""
        if isinstance(data, str):
            # Check if it's a file path
            if Path(data).exists():
                return "file"
            
            # Check if it's a URL
            if re.match(r'https?://', data):
                return "url"
            
            # Check if it's JSON
            try:
                json.loads(data)
                return "json"
            except:
                pass
            
            # Check if it's CSV-like
            if '\n' in data and (',' in data or '\t' in data):
                return "csv"
            
            # Default to text
            return "text"
        
        elif isinstance(data, (list, dict)):
            return "json"
        
        elif isinstance(data, (int, float, np.number)):
            return "numeric"
        
        elif isinstance(data, pd.DataFrame):
            return "csv"
        
        else:
            return "unknown"
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content for insights"""
        try:
            blob = TextBlob(text)
            
            # Basic statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(blob.sentences)
            
            # Sentiment analysis
            sentiment = blob.sentiment
            
            # Extract key phrases (simple noun phrases)
            noun_phrases = list(blob.noun_phrases)[:10]
            
            # Find patterns
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s\d{3}-\d{4}\b'
            url_pattern = r'https?://[^\s]+'
            
            emails = re.findall(email_pattern, text)
            phones = re.findall(phone_pattern, text)
            urls = re.findall(url_pattern, text)
            
            # Generate insights
            insights = []
            
            if sentiment.polarity > 0.1:
                insights.append(f"Text has positive sentiment (polarity: {sentiment.polarity:.2f})")
            elif sentiment.polarity < -0.1:
                insights.append(f"Text has negative sentiment (polarity: {sentiment.polarity:.2f})")
            else:
                insights.append(f"Text has neutral sentiment (polarity: {sentiment.polarity:.2f})")
            
            if word_count > 1000:
                insights.append("This is a long document that might contain detailed information")
            
            if emails:
                insights.append(f"Found {len(emails)} email addresses")
            
            if phones:
                insights.append(f"Found {len(phones)} phone numbers")
            
            if urls:
                insights.append(f"Found {len(urls)} URLs")
            
            if noun_phrases:
                insights.append(f"Key topics include: {', '.join(noun_phrases[:5])}")
            
            return {
                "status": "success",
                "data_type": "text",
                "analysis": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "sentence_count": sentence_count,
                    "sentiment": {
                        "polarity": sentiment.polarity,
                        "subjectivity": sentiment.subjectivity
                    },
                    "key_phrases": noun_phrases,
                    "extracted_data": {
                        "emails": emails,
                        "phones": phones,
                        "urls": urls
                    }
                },
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_csv(self, data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze CSV/tabular data"""
        try:
            if isinstance(data, str):
                # Try to parse as CSV
                try:
                    df = pd.read_csv(pd.StringIO(data))
                except:
                    # Try tab-separated
                    df = pd.read_csv(pd.StringIO(data), sep='\t')
            else:
                df = data
            
            # Basic statistics
            shape = df.shape
            columns = list(df.columns)
            dtypes = df.dtypes.to_dict()
            
            # Missing data analysis
            missing_data = df.isnull().sum().to_dict()
            
            # Numeric column analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_stats = {}
            for col in numeric_cols:
                numeric_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
            
            # Categorical analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            categorical_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(5)
                categorical_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_common": value_counts.to_dict()
                }
            
            # Generate insights
            insights = []
            insights.append(f"Dataset contains {shape[0]} rows and {shape[1]} columns")
            
            if missing_data and any(missing_data.values()):
                missing_cols = [col for col, count in missing_data.items() if count > 0]
                insights.append(f"Missing data found in columns: {', '.join(missing_cols)}")
            
            if numeric_cols.any():
                insights.append(f"Numeric columns available for analysis: {', '.join(numeric_cols)}")
            
            if categorical_cols.any():
                insights.append(f"Categorical columns: {', '.join(categorical_cols)}")
            
            # Detect potential relationships
            if len(numeric_cols) >= 2:
                insights.append("Multiple numeric columns detected - correlation analysis possible")
            
            return {
                "status": "success",
                "data_type": "csv",
                "analysis": {
                    "shape": shape,
                    "columns": columns,
                    "data_types": {k: str(v) for k, v in dtypes.items()},
                    "missing_data": missing_data,
                    "numeric_statistics": numeric_stats,
                    "categorical_statistics": categorical_stats
                },
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing CSV data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_json(self, data: Union[str, dict, list]) -> Dict[str, Any]:
        """Analyze JSON data structure"""
        try:
            if isinstance(data, str):
                json_data = json.loads(data)
            else:
                json_data = data
            
            # Analyze structure
            def analyze_structure(obj, path="root"):
                if isinstance(obj, dict):
                    return {
                        "type": "object",
                        "keys": list(obj.keys()),
                        "key_count": len(obj),
                        "nested_structure": {k: analyze_structure(v, f"{path}.{k}") for k, v in obj.items()}
                    }
                elif isinstance(obj, list):
                    return {
                        "type": "array",
                        "length": len(obj),
                        "sample_items": [analyze_structure(item, f"{path}[{i}]") for i, item in enumerate(obj[:3])]
                    }
                else:
                    return {
                        "type": type(obj).__name__,
                        "value": str(obj)[:100] if len(str(obj)) > 100 else str(obj)
                    }
            
            structure = analyze_structure(json_data)
            
            # Generate insights
            insights = []
            
            if isinstance(json_data, dict):
                insights.append(f"JSON object with {len(json_data)} top-level keys")
                if "id" in json_data:
                    insights.append("Contains ID field - might be a record or entity")
                if "timestamp" in json_data or "date" in json_data:
                    insights.append("Contains timestamp/date information")
            elif isinstance(json_data, list):
                insights.append(f"JSON array with {len(json_data)} items")
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    insights.append("Array of objects - suitable for tabular analysis")
            
            # Check for nested data
            json_str = json.dumps(json_data)
            nesting_level = json_str.count('{') + json_str.count('[')
            if nesting_level > 5:
                insights.append("Highly nested structure detected")
            
            return {
                "status": "success",
                "data_type": "json",
                "analysis": {
                    "structure": structure,
                    "size_bytes": len(json.dumps(json_data)),
                    "nesting_level": nesting_level
                },
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing JSON data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_numeric(self, data: Union[int, float, List[Union[int, float]]]) -> Dict[str, Any]:
        """Analyze numeric data"""
        try:
            if isinstance(data, (int, float)):
                data = [data]
            
            arr = np.array(data)
            
            analysis = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "range": float(np.max(arr) - np.min(arr))
            }
            
            insights = []
            insights.append(f"Numeric data with {len(arr)} values")
            
            if analysis["std"] < 0.1 * abs(analysis["mean"]):
                insights.append("Low variance detected - values are similar")
            elif analysis["std"] > analysis["mean"]:
                insights.append("High variance detected - values are spread out")
            
            if analysis["min"] == analysis["max"]:
                insights.append("All values are identical")
            
            return {
                "status": "success",
                "data_type": "numeric",
                "analysis": analysis,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing numeric data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze file based on its type"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"status": "error", "message": "File does not exist"}
            
            # Get file info
            file_size = path.stat().st_size
            mime_type, _ = mimetypes.guess_type(file_path)
            
            insights = []
            insights.append(f"File size: {file_size} bytes")
            
            if mime_type:
                insights.append(f"MIME type: {mime_type}")
            
            # Analyze based on file extension
            suffix = path.suffix.lower()
            
            if suffix in ['.txt', '.md', '.py', '.js', '.html', '.css']:
                # Text-based file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return await self._analyze_text(content)
            
            elif suffix in ['.csv', '.tsv']:
                # CSV file
                df = pd.read_csv(file_path)
                return await self._analyze_csv(df)
            
            elif suffix == '.json':
                # JSON file
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                return await self._analyze_json(json_data)
            
            else:
                return {
                    "status": "success",
                    "data_type": "file",
                    "analysis": {
                        "file_path": str(path),
                        "file_size": file_size,
                        "mime_type": mime_type,
                        "extension": suffix
                    },
                    "insights": insights,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing file: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_image(self, image_data: Any) -> Dict[str, Any]:
        """Placeholder for image analysis"""
        return {
            "status": "success",
            "data_type": "image",
            "analysis": {"message": "Image analysis not yet implemented"},
            "insights": ["Image data detected - visual analysis capabilities coming soon"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_url(self, url: str) -> Dict[str, Any]:
        """Placeholder for URL analysis"""
        return {
            "status": "success",
            "data_type": "url",
            "analysis": {"url": url},
            "insights": [f"URL detected: {url} - web scraping capabilities coming soon"],
            "timestamp": datetime.now().isoformat()
        }