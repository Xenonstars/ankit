#!/usr/bin/env python3
"""
Simplified Personal Work Agent Server

A minimal version that can run without heavy ML dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Personal Work Agent API",
    description="An intelligent agent that learns from your work data and provides assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
knowledge_base = []
interactions = []

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    context_limit: Optional[int] = 5

class TrainingRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    data: Any
    data_type: Optional[str] = "auto"

class AgentResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    timestamp: datetime

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_name": "WorkAssistant",
        "knowledge_base_size": len(knowledge_base)
    }

# Simple query endpoint
@app.post("/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """Query the agent with a question"""
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Simple keyword matching for demo
        question_lower = request.question.lower()
        relevant_docs = []
        
        for item in knowledge_base:
            if any(word in item['content'].lower() for word in question_lower.split()):
                relevant_docs.append(item['content'][:100])
        
        if relevant_docs:
            response_text = f"Based on your knowledge base, I found {len(relevant_docs)} relevant items. Here's what I know: {' '.join(relevant_docs[:2])}"
            confidence = min(0.8, len(relevant_docs) / 5.0)
        else:
            response_text = "I don't have enough information in my knowledge base to answer that question. Please train me with relevant data first."
            confidence = 0.1
        
        agent_response = AgentResponse(
            response=response_text,
            confidence=confidence,
            sources=relevant_docs[:3],
            timestamp=datetime.now()
        )
        
        # Store interaction
        interactions.append({
            "question": request.question,
            "response": response_text,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        return agent_response
        
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Training endpoint
@app.post("/train")
async def train_agent(request: TrainingRequest):
    """Train the agent with new data"""
    try:
        logger.info(f"Training with content length: {len(request.content)}")
        
        # Simple storage
        knowledge_item = {
            "id": len(knowledge_base) + 1,
            "content": request.content,
            "metadata": request.metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        knowledge_base.append(knowledge_item)
        
        return {
            "success": True,
            "message": f"Successfully added knowledge item {knowledge_item['id']}",
            "knowledge_base_size": len(knowledge_base),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis endpoint
@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze data (simplified version)"""
    try:
        data = request.data
        data_type = request.data_type
        
        # Simple analysis
        insights = []
        
        if isinstance(data, str):
            word_count = len(data.split())
            char_count = len(data)
            insights.append(f"Text contains {word_count} words and {char_count} characters")
            
            if "meeting" in data.lower():
                insights.append("This appears to be meeting-related content")
            if "task" in data.lower() or "todo" in data.lower():
                insights.append("This appears to contain task information")
                
        elif isinstance(data, dict):
            key_count = len(data.keys())
            insights.append(f"JSON object with {key_count} top-level keys")
            
            if "id" in data:
                insights.append("Contains ID field - might be a record")
            if any(key in str(data).lower() for key in ["date", "time"]):
                insights.append("Contains temporal information")
                
        elif isinstance(data, list):
            insights.append(f"Array with {len(data)} items")
            
        return {
            "status": "success",
            "data_type": data_type,
            "analysis": {
                "type": type(data).__name__,
                "size": len(str(data)),
                "insights": insights
            },
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Stats endpoint
@app.get("/stats")
async def get_stats():
    """Get agent statistics"""
    return {
        "knowledge_base_size": len(knowledge_base),
        "total_interactions": len(interactions),
        "agent_name": "WorkAssistant",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

# Recent interactions
@app.get("/memory/recent")
async def get_recent_interactions(limit: int = 10):
    """Get recent interactions"""
    return {
        "interactions": interactions[-limit:] if interactions else [],
        "total": len(interactions)
    }

# Web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve a simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Personal Work Agent</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; padding: 20px; background: #f5f5f5; 
            }
            .container { 
                max-width: 800px; margin: 0 auto; background: white; 
                padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header { text-align: center; margin-bottom: 30px; }
            .section { 
                margin: 20px 0; padding: 20px; background: #f8f9fa; 
                border-radius: 8px; border-left: 4px solid #007bff;
            }
            .input-group { margin: 15px 0; }
            label { display: block; font-weight: 600; margin-bottom: 5px; }
            input, textarea { 
                width: 100%; padding: 10px; border: 1px solid #ddd; 
                border-radius: 6px; font-size: 14px; 
            }
            button { 
                background: #007bff; color: white; border: none; 
                padding: 12px 24px; border-radius: 6px; cursor: pointer;
                font-size: 14px; font-weight: 600;
            }
            button:hover { background: #0056b3; }
            .response { 
                background: #e9ecef; padding: 15px; border-radius: 6px; 
                margin: 15px 0; border-left: 4px solid #28a745;
            }
            .stats { display: flex; gap: 20px; }
            .stat-box { 
                flex: 1; text-align: center; padding: 15px; 
                background: white; border-radius: 6px; border: 1px solid #ddd;
            }
            .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
            .api-links { margin-top: 20px; }
            .api-links a { 
                display: inline-block; margin: 5px 10px 5px 0; 
                padding: 8px 16px; background: #6c757d; color: white; 
                text-decoration: none; border-radius: 4px; font-size: 12px;
            }
            .api-links a:hover { background: #545b62; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Personal Work Agent</h1>
                <p>Your intelligent assistant for work data analysis and support</p>
            </div>
            
            <div id="stats" class="stats">
                <div class="stat-box">
                    <div class="stat-number" id="knowledge-count">0</div>
                    <div class="stat-label">Knowledge Items</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="interaction-count">0</div>
                    <div class="stat-label">Interactions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">Active</div>
                    <div class="stat-label">Status</div>
                </div>
            </div>
            
            <div class="section">
                <h3>💬 Ask a Question</h3>
                <div class="input-group">
                    <label for="question">What would you like to know?</label>
                    <input type="text" id="question" placeholder="e.g., What are my current tasks?" />
                </div>
                <button onclick="askQuestion()">Ask Question</button>
                <div id="query-response" class="response" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h3>📚 Train the Agent</h3>
                <div class="input-group">
                    <label for="training-content">Add knowledge (meeting notes, tasks, etc.)</label>
                    <textarea id="training-content" rows="4" placeholder="Enter your work data here..."></textarea>
                </div>
                <button onclick="trainAgent()">Add to Knowledge Base</button>
                <div id="training-response" class="response" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h3>🔗 API Documentation</h3>
                <p>Access the full API capabilities:</p>
                <div class="api-links">
                    <a href="/docs" target="_blank">Interactive API Docs</a>
                    <a href="/redoc" target="_blank">Alternative Docs</a>
                    <a href="/health" target="_blank">Health Check</a>
                    <a href="/stats" target="_blank">Statistics</a>
                </div>
            </div>
        </div>
        
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    document.getElementById('knowledge-count').textContent = stats.knowledge_base_size;
                    document.getElementById('interaction-count').textContent = stats.total_interactions;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const responseDiv = document.getElementById('query-response');
                
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: question})
                    });
                    
                    const result = await response.json();
                    responseDiv.innerHTML = `
                        <strong>Answer:</strong> ${result.response}<br>
                        <small>Confidence: ${(result.confidence * 100).toFixed(1)}%</small>
                    `;
                    responseDiv.style.display = 'block';
                    loadStats();
                } catch (error) {
                    responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                    responseDiv.style.display = 'block';
                }
            }
            
            async function trainAgent() {
                const content = document.getElementById('training-content').value;
                const responseDiv = document.getElementById('training-response');
                
                if (!content.trim()) {
                    alert('Please enter some content to train the agent');
                    return;
                }
                
                try {
                    const response = await fetch('/train', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            content: content,
                            metadata: {type: 'manual_entry', timestamp: new Date().toISOString()}
                        })
                    });
                    
                    const result = await response.json();
                    responseDiv.innerHTML = `<strong>Success:</strong> ${result.message}`;
                    responseDiv.style.display = 'block';
                    document.getElementById('training-content').value = '';
                    loadStats();
                } catch (error) {
                    responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
                    responseDiv.style.display = 'block';
                }
            }
            
            // Load stats on page load
            loadStats();
            
            // Allow Enter key to submit question
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    
    print("🤖 Starting Personal Work Agent (Simplified Version)...")
    print("🌐 Server: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("💾 This version uses in-memory storage for demo purposes")
    print("🚀 Starting server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )