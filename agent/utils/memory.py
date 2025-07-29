import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from .config import Config


class MemoryManager:
    """
    Manages the agent's memory including interactions, training sessions, and analytics
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.db_path = self._get_db_path()
        self._init_database()
    
    def _get_db_path(self) -> str:
        """Get database path from config"""
        if "sqlite" in self.config.database_url:
            return self.config.database_url.replace("sqlite:///", "")
        else:
            # Fallback to local SQLite
            return "./data/memory.db"
    
    def _init_database(self):
        """Initialize the memory database"""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    confidence REAL,
                    sources TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Create training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_length INTEGER,
                    chunks_created INTEGER,
                    content_type TEXT,
                    source_info TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER,
                    rating INTEGER,
                    feedback_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions (id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Memory database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing memory database: {str(e)}")
    
    async def add_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Add a new interaction to memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO interactions (question, response, confidence, sources, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_data.get("question", ""),
                interaction_data.get("response", ""),
                interaction_data.get("confidence", 0.0),
                json.dumps(interaction_data.get("sources", [])),
                interaction_data.get("session_id", ""),
                json.dumps(interaction_data.get("metadata", {}))
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug("Interaction added to memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding interaction to memory: {str(e)}")
            return False
    
    async def add_training_session(self, training_data: Dict[str, Any]) -> bool:
        """Add a training session record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO training_sessions (content_length, chunks_created, content_type, source_info, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                training_data.get("content_length", 0),
                training_data.get("chunks_created", 0),
                training_data.get("content_type", "unknown"),
                json.dumps(training_data.get("source_info", {})),
                json.dumps(training_data.get("metadata", {}))
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug("Training session added to memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding training session to memory: {str(e)}")
            return False
    
    async def get_recent_interactions(self, limit: int = 10, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM interactions 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM interactions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            interactions = []
            for row in rows:
                interactions.append({
                    "id": row[0],
                    "question": row[1],
                    "response": row[2],
                    "confidence": row[3],
                    "sources": json.loads(row[4]) if row[4] else [],
                    "timestamp": row[5],
                    "session_id": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                })
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error getting recent interactions: {str(e)}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get interaction stats
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM interactions 
                WHERE timestamp > datetime('now', '-7 days')
            """)
            recent_interactions = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(confidence) FROM interactions WHERE confidence > 0")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Get training stats
            cursor.execute("SELECT COUNT(*) FROM training_sessions")
            total_training_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(chunks_created) FROM training_sessions")
            total_chunks = cursor.fetchone()[0] or 0
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM training_sessions 
                WHERE timestamp > datetime('now', '-7 days')
            """)
            recent_training = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_interactions": total_interactions,
                "recent_interactions": recent_interactions,
                "average_confidence": round(avg_confidence, 3),
                "total_training_sessions": total_training_sessions,
                "total_chunks_processed": total_chunks,
                "recent_training_sessions": recent_training,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}
    
    async def search_interactions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through interaction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple text search in questions and responses
            cursor.execute("""
                SELECT * FROM interactions 
                WHERE question LIKE ? OR response LIKE ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            interactions = []
            for row in rows:
                interactions.append({
                    "id": row[0],
                    "question": row[1],
                    "response": row[2],
                    "confidence": row[3],
                    "timestamp": row[5],
                    "relevance": "text_match"  # Could be enhanced with semantic search
                })
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error searching interactions: {str(e)}")
            return []
    
    async def add_feedback(self, interaction_id: int, rating: int, feedback_text: str = "") -> bool:
        """Add feedback for an interaction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (interaction_id, rating, feedback_text)
                VALUES (?, ?, ?)
            """, (interaction_id, rating, feedback_text))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Feedback added for interaction {interaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    async def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get confidence trends
            cursor.execute("""
                SELECT DATE(timestamp) as date, AVG(confidence) as avg_confidence, COUNT(*) as count
                FROM interactions 
                WHERE timestamp > datetime('now', '-{} days') AND confidence > 0
                GROUP BY DATE(timestamp)
                ORDER BY date
            """.format(days))
            
            confidence_trends = []
            for row in cursor.fetchall():
                confidence_trends.append({
                    "date": row[0],
                    "avg_confidence": round(row[1], 3),
                    "interaction_count": row[2]
                })
            
            # Get feedback summary
            cursor.execute("""
                SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
                FROM feedback 
                WHERE timestamp > datetime('now', '-{} days')
            """.format(days))
            
            feedback_row = cursor.fetchone()
            avg_rating = feedback_row[0] if feedback_row[0] else 0.0
            total_feedback = feedback_row[1]
            
            # Get training frequency
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as sessions, SUM(chunks_created) as chunks
                FROM training_sessions 
                WHERE timestamp > datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """.format(days))
            
            training_trends = []
            for row in cursor.fetchall():
                training_trends.append({
                    "date": row[0],
                    "training_sessions": row[1],
                    "chunks_created": row[2]
                })
            
            conn.close()
            
            return {
                "confidence_trends": confidence_trends,
                "average_user_rating": round(avg_rating, 2),
                "total_feedback_count": total_feedback,
                "training_trends": training_trends,
                "period_days": days
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {"error": str(e)}
    
    async def clear_old_data(self, days_to_keep: int = 90) -> bool:
        """Clear old data to manage database size"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear old interactions
            cursor.execute("""
                DELETE FROM interactions 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            interactions_deleted = cursor.rowcount
            
            # Clear old training sessions
            cursor.execute("""
                DELETE FROM training_sessions 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            training_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleared {interactions_deleted} old interactions and {training_deleted} old training sessions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing old data: {str(e)}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all memory data (use with caution)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM interactions")
            cursor.execute("DELETE FROM training_sessions")
            cursor.execute("DELETE FROM analytics")
            cursor.execute("DELETE FROM feedback")
            
            conn.commit()
            conn.close()
            
            self.logger.info("All memory data cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing all data: {str(e)}")
            return False