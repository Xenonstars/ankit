import re
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

from ..utils.config import Config


class UnstructuredTrainer:
    """
    Processes and prepares unstructured work data for training the agent
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Initialize spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def process_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process unstructured content and create training chunks
        """
        self.logger.info(f"Processing content of length: {len(content)}")
        
        # Clean and preprocess the content
        cleaned_content = self._clean_content(content)
        
        # Extract metadata from content
        extracted_metadata = self._extract_metadata(cleaned_content)
        
        # Combine provided metadata with extracted metadata
        combined_metadata = {**metadata, **extracted_metadata}
        
        # Create chunks
        chunks = self._create_chunks(cleaned_content, combined_metadata)
        
        # Enhance chunks with additional processing
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = await self._enhance_chunk(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        self.logger.info(f"Created {len(enhanced_chunks)} chunks from content")
        return enhanced_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep punctuation
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', '', content)
        
        # Normalize line breaks
        content = content.replace('\n', ' ').replace('\r', ' ')
        
        return content.strip()
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {}
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b',   # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        
        if dates:
            metadata['contains_dates'] = True
            metadata['extracted_dates'] = dates[:5]  # Limit to first 5
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            metadata['contains_emails'] = True
            metadata['email_count'] = len(emails)
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s\d{3}-\d{4}\b'
        phones = re.findall(phone_pattern, content)
        if phones:
            metadata['contains_phones'] = True
            metadata['phone_count'] = len(phones)
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content)
        if urls:
            metadata['contains_urls'] = True
            metadata['url_count'] = len(urls)
        
        # Extract project/task indicators
        project_indicators = [
            r'\bproject\s+[a-zA-Z0-9_-]+\b',
            r'\btask\s+[a-zA-Z0-9_-]+\b',
            r'\bticket\s+[a-zA-Z0-9_-]+\b',
            r'\bissue\s+[a-zA-Z0-9_-]+\b'
        ]
        
        projects = []
        for pattern in project_indicators:
            projects.extend(re.findall(pattern, content, re.IGNORECASE))
        
        if projects:
            metadata['contains_project_refs'] = True
            metadata['project_references'] = projects[:3]
        
        # Detect content type based on keywords
        if any(word in content.lower() for word in ['meeting', 'agenda', 'minutes', 'discussed']):
            metadata['content_type'] = 'meeting'
        elif any(word in content.lower() for word in ['todo', 'task', 'deadline', 'due']):
            metadata['content_type'] = 'task'
        elif any(word in content.lower() for word in ['email', 'subject', 'from:', 'to:']):
            metadata['content_type'] = 'email'
        elif any(word in content.lower() for word in ['note', 'remember', 'important']):
            metadata['content_type'] = 'note'
        else:
            metadata['content_type'] = 'general'
        
        return metadata
    
    def _create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create manageable chunks from content"""
        chunks = []
        
        # Strategy 1: Split by sentences for better semantic coherence
        sentences = sent_tokenize(content)
        
        current_chunk = ""
        current_token_count = 0
        max_tokens = self.config.max_chunk_tokens
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed the limit, save current chunk
            if current_token_count + sentence_tokens > max_tokens and current_chunk:
                chunk = self._create_chunk_dict(current_chunk, metadata)
                chunks.append(chunk)
                current_chunk = sentence
                current_token_count = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_token_count += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = self._create_chunk_dict(current_chunk, metadata)
            chunks.append(chunk)
        
        # If no chunks were created (content too small), create one chunk
        if not chunks:
            chunk = self._create_chunk_dict(content, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_dict(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized chunk dictionary"""
        chunk_id = hashlib.md5(text.encode()).hexdigest()
        
        return {
            "id": f"chunk_{chunk_id}_{datetime.now().timestamp()}",
            "text": text.strip(),
            "token_count": self._count_tokens(text),
            "word_count": len(text.split()),
            "metadata": {
                **metadata,
                "chunk_created_at": datetime.now().isoformat(),
                "chunk_length": len(text)
            }
        }
    
    async def _enhance_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance chunk with additional NLP processing"""
        text = chunk["text"]
        
        # Add named entity recognition if spaCy is available
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                if entities:
                    chunk["metadata"]["entities"] = entities[:10]  # Limit to 10
                
                # Extract key phrases (noun phrases)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                if noun_phrases:
                    chunk["metadata"]["key_phrases"] = noun_phrases[:5]  # Limit to 5
                
            except Exception as e:
                self.logger.warning(f"Error in NLP processing: {str(e)}")
        
        # Add keyword extraction
        keywords = self._extract_keywords(text)
        if keywords:
            chunk["metadata"]["keywords"] = keywords
        
        # Add sentiment if available
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            chunk["metadata"]["sentiment"] = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
        except:
            pass
        
        return chunk
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction based on frequency and filtering
        words = word_tokenize(text.lower())
        
        # Filter out stopwords and short words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        keywords = [
            word for word in words 
            if word.isalpha() and len(word) > 3 and word not in stop_words
        ]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Return top keywords
        return [word for word, freq in word_freq.most_common(10)]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate token count
            return len(text.split()) * 1.3  # Rough approximation
    
    async def process_file_content(self, file_path: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Process content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_metadata = {
                "source_file": file_path,
                "file_type": file_path.split('.')[-1] if '.' in file_path else 'unknown'
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            return await self.process_content(content, file_metadata)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    async def process_structured_data(self, data: Dict[str, Any], context: str = "") -> List[Dict[str, Any]]:
        """Process structured data (like JSON) into training chunks"""
        # Convert structured data to text representation
        text_content = self._structured_to_text(data, context)
        
        metadata = {
            "data_type": "structured",
            "context": context,
            "original_keys": list(data.keys()) if isinstance(data, dict) else []
        }
        
        return await self.process_content(text_content, metadata)
    
    def _structured_to_text(self, data: Any, context: str = "", level: int = 0) -> str:
        """Convert structured data to readable text"""
        if level > 3:  # Prevent deep recursion
            return str(data)
        
        if isinstance(data, dict):
            parts = []
            for key, value in data.items():
                value_text = self._structured_to_text(value, f"{context}.{key}", level + 1)
                parts.append(f"{key}: {value_text}")
            return "; ".join(parts)
        
        elif isinstance(data, list):
            if len(data) <= 5:
                items = [self._structured_to_text(item, f"{context}[{i}]", level + 1) for i, item in enumerate(data)]
                return ", ".join(items)
            else:
                # For long lists, sample first few items
                items = [self._structured_to_text(item, f"{context}[{i}]", level + 1) for i, item in enumerate(data[:3])]
                return f"{', '.join(items)} (and {len(data) - 3} more items)"
        
        else:
            return str(data)