import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

import chromadb
from sentence_transformers import SentenceTransformer
import openai
from pydantic import BaseModel

from ..data.analyzer import DataAnalyzer
from ..training.trainer import UnstructuredTrainer
from ..utils.memory import MemoryManager
from ..utils.config import Config


class AgentResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    timestamp: datetime


class PersonalWorkAgent:
    """
    An intelligent agent that analyzes data, learns from unstructured work information,
    and provides assistance based on learned knowledge.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_analyzer = DataAnalyzer(config)
        self.trainer = UnstructuredTrainer(config)
        self.memory_manager = MemoryManager(config)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path=config.vector_db_path)
        self.collection = self._get_or_create_collection()
        
        # Initialize OpenAI client
        openai.api_key = config.openai_api_key
        
        self.logger.info(f"Personal Work Agent '{config.agent_name}' initialized")
    
    def _get_or_create_collection(self):
        """Get or create the main knowledge collection"""
        try:
            return self.chroma_client.get_collection(name="work_knowledge")
        except:
            return self.chroma_client.create_collection(
                name="work_knowledge",
                metadata={"description": "Personal work knowledge base"}
            )
    
    async def analyze_data(self, data: Any, data_type: str = "auto") -> Dict[str, Any]:
        """
        Analyze any type of data fed to the agent
        """
        self.logger.info(f"Analyzing data of type: {data_type}")
        
        try:
            # Use data analyzer to process the input
            analysis_result = await self.data_analyzer.analyze(data, data_type)
            
            # Store insights in memory
            if analysis_result.get("insights"):
                await self._store_insights(analysis_result["insights"], "data_analysis")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def train_on_data(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Train the agent on unstructured work data
        """
        self.logger.info("Training on new unstructured data")
        
        try:
            # Process and chunk the content
            chunks = await self.trainer.process_content(content, metadata or {})
            
            # Generate embeddings and store in vector database
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk["text"])
                
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]],
                    ids=[chunk["id"]]
                )
            
            # Update memory with training session
            await self.memory_manager.add_training_session({
                "content_length": len(content),
                "chunks_created": len(chunks),
                "timestamp": datetime.now(),
                "metadata": metadata
            })
            
            self.logger.info(f"Successfully trained on {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training on data: {str(e)}")
            return False
    
    async def query(self, question: str, context_limit: int = 5) -> AgentResponse:
        """
        Query the agent with a question and get an intelligent response
        """
        self.logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode(question)
            
            # Search for relevant context in vector database
            results = self.collection.query(
                query_embeddings=[question_embedding.tolist()],
                n_results=context_limit
            )
            
            # Build context from retrieved documents
            context_docs = results["documents"][0] if results["documents"] else []
            context = "\n\n".join(context_docs)
            
            # Generate response using OpenAI
            response = await self._generate_response(question, context)
            
            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(results)
            
            # Prepare sources
            sources = [doc[:100] + "..." for doc in context_docs[:3]]
            
            agent_response = AgentResponse(
                response=response,
                confidence=confidence,
                sources=sources,
                timestamp=datetime.now()
            )
            
            # Store interaction in memory
            await self.memory_manager.add_interaction({
                "question": question,
                "response": response,
                "confidence": confidence,
                "timestamp": datetime.now()
            })
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return AgentResponse(
                response=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                sources=[],
                timestamp=datetime.now()
            )
    
    async def _generate_response(self, question: str, context: str) -> str:
        """Generate response using OpenAI with context"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are {self.config.agent_name}, a personal work assistant trained on the user's work data. 
                        Use the provided context to answer questions accurately. If the context doesn't contain relevant information, 
                        say so honestly. Be helpful, concise, and reference specific details from the context when possible."""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I couldn't generate a response at this time."
    
    def _calculate_confidence(self, search_results: Dict) -> float:
        """Calculate confidence score based on search results"""
        if not search_results.get("distances") or not search_results["distances"][0]:
            return 0.0
        
        # Convert distances to similarities (lower distance = higher similarity)
        distances = search_results["distances"][0]
        similarities = [1 - min(d, 1.0) for d in distances]
        
        # Return average similarity as confidence
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    async def _store_insights(self, insights: List[str], source: str):
        """Store analysis insights in the knowledge base"""
        for insight in insights:
            embedding = self.embedding_model.encode(insight)
            
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[insight],
                metadatas=[{
                    "type": "insight",
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"insight_{datetime.now().timestamp()}_{hash(insight)}"]
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics and performance metrics"""
        collection_count = self.collection.count()
        memory_stats = await self.memory_manager.get_stats()
        
        return {
            "knowledge_base_size": collection_count,
            "memory_stats": memory_stats,
            "agent_name": self.config.agent_name,
            "embedding_model": self.config.embedding_model,
            "status": "active"
        }
    
    async def reset_knowledge(self) -> bool:
        """Reset the agent's knowledge base (use with caution)"""
        try:
            self.chroma_client.delete_collection("work_knowledge")
            self.collection = self._get_or_create_collection()
            await self.memory_manager.clear_all()
            self.logger.info("Knowledge base reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting knowledge base: {str(e)}")
            return False