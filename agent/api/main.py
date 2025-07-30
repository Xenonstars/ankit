from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import json
import os

from ..core.agent import PersonalWorkAgent, AgentResponse
from ..utils.config import Config
from ..utils.memory import MemoryManager

# Initialize configuration
config = Config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)

# Initialize the agent
agent = PersonalWorkAgent(config)

# Initialize FastAPI app
app = FastAPI(
    title="Personal Work Agent API",
    description="An intelligent agent that learns from your work data and provides assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("agent/web/static"):
    app.mount("/static", StaticFiles(directory="agent/web/static"), name="static")


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    context_limit: Optional[int] = 5
    session_id: Optional[str] = None


class TrainingRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    data: Any
    data_type: Optional[str] = "auto"


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: int  # 1-5 scale
    feedback_text: Optional[str] = ""


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_name": config.agent_name
    }


# Agent query endpoint
@app.post("/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """Query the agent with a question"""
    try:
        response = await agent.query(request.question, request.context_limit)
        return response
    except Exception as e:
        logging.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoint
@app.post("/train")
async def train_agent(request: TrainingRequest):
    """Train the agent with new data"""
    try:
        success = await agent.train_on_data(request.content, request.metadata)
        return {
            "success": success,
            "message": "Training completed successfully" if success else "Training failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error in training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# File upload and training endpoint
@app.post("/train/file")
async def train_from_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Train the agent from an uploaded file"""
    try:
        # Save uploaded file
        upload_path = f"{config.upload_directory}/{file.filename}"
        os.makedirs(config.upload_directory, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read and process file content
        try:
            with open(upload_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(upload_path, 'r', encoding='latin-1') as f:
                file_content = f.read()
        
        # Parse metadata if provided
        file_metadata = json.loads(metadata) if metadata else {}
        file_metadata.update({
            "filename": file.filename,
            "file_size": len(content),
            "upload_timestamp": datetime.now().isoformat()
        })
        
        # Train the agent
        success = await agent.train_on_data(file_content, file_metadata)
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return {
            "success": success,
            "filename": file.filename,
            "file_size": len(content),
            "message": "File processed successfully" if success else "File processing failed"
        }
        
    except Exception as e:
        logging.error(f"Error in file training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data analysis endpoint
@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze data fed to the agent"""
    try:
        analysis = await agent.analyze_data(request.data, request.data_type)
        return analysis
    except Exception as e:
        logging.error(f"Error in analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get agent statistics
@app.get("/stats")
async def get_agent_stats():
    """Get agent statistics and performance metrics"""
    try:
        stats = await agent.get_stats()
        return stats
    except Exception as e:
        logging.error(f"Error in stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory endpoints
@app.get("/memory/recent")
async def get_recent_interactions(limit: int = 10, session_id: Optional[str] = None):
    """Get recent interactions"""
    try:
        interactions = await agent.memory_manager.get_recent_interactions(limit, session_id)
        return {"interactions": interactions}
    except Exception as e:
        logging.error(f"Error getting recent interactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search")
async def search_interactions(query: str, limit: int = 5):
    """Search through interaction history"""
    try:
        results = await agent.memory_manager.search_interactions(query, limit)
        return {"results": results}
    except Exception as e:
        logging.error(f"Error searching interactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """Add feedback for an interaction"""
    try:
        success = await agent.memory_manager.add_feedback(
            request.interaction_id,
            request.rating,
            request.feedback_text
        )
        return {
            "success": success,
            "message": "Feedback added successfully" if success else "Failed to add feedback"
        }
    except Exception as e:
        logging.error(f"Error adding feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/metrics")
async def get_performance_metrics(days: int = 30):
    """Get performance metrics"""
    try:
        metrics = await agent.memory_manager.get_performance_metrics(days)
        return metrics
    except Exception as e:
        logging.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent management endpoints
@app.post("/agent/reset")
async def reset_agent():
    """Reset the agent's knowledge base (use with caution)"""
    try:
        success = await agent.reset_knowledge()
        return {
            "success": success,
            "message": "Agent knowledge base reset" if success else "Reset failed"
        }
    except Exception as e:
        logging.error(f"Error resetting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get agent configuration (sensitive data excluded)"""
    try:
        return config.to_dict()
    except Exception as e:
        logging.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations
@app.post("/train/batch")
async def batch_train(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    """Train the agent from multiple files"""
    try:
        results = []
        base_metadata = json.loads(metadata) if metadata else {}
        
        for file in files:
            try:
                # Process each file
                content = await file.read()
                file_content = content.decode('utf-8', errors='ignore')
                
                file_metadata = base_metadata.copy()
                file_metadata.update({
                    "filename": file.filename,
                    "file_size": len(content),
                    "batch_upload": True,
                    "upload_timestamp": datetime.now().isoformat()
                })
                
                success = await agent.train_on_data(file_content, file_metadata)
                
                results.append({
                    "filename": file.filename,
                    "success": success,
                    "file_size": len(content)
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        successful_uploads = sum(1 for r in results if r["success"])
        
        return {
            "total_files": len(files),
            "successful_uploads": successful_uploads,
            "failed_uploads": len(files) - successful_uploads,
            "results": results
        }
        
    except Exception as e:
        logging.error(f"Error in batch training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Web interface endpoint
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface"""
    try:
        # Check if web interface exists
        web_interface_path = "agent/web/templates/index.html"
        if os.path.exists(web_interface_path):
            with open(web_interface_path, 'r') as f:
                return f.read()
        else:
            # Return a simple default interface
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Personal Work Agent</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                    button { padding: 10px 20px; margin: 5px; }
                    textarea { width: 100%; height: 100px; margin: 10px 0; }
                    .response { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Personal Work Agent</h1>
                    <p>Your intelligent assistant for work data analysis and support.</p>
                    
                    <div class="section">
                        <h3>API Endpoints Available:</h3>
                        <ul>
                            <li><strong>POST /query</strong> - Ask questions to your agent</li>
                            <li><strong>POST /train</strong> - Train with text data</li>
                            <li><strong>POST /train/file</strong> - Train with file uploads</li>
                            <li><strong>POST /analyze</strong> - Analyze data</li>
                            <li><strong>GET /stats</strong> - Get agent statistics</li>
                            <li><strong>GET /memory/recent</strong> - Get recent interactions</li>
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h3>Quick Start:</h3>
                        <p>1. Use the API endpoints to interact with your agent</p>
                        <p>2. Train it with your work data using /train or /train/file</p>
                        <p>3. Ask questions using /query to get intelligent responses</p>
                        <p>4. Monitor performance with /stats and /memory/metrics</p>
                    </div>
                    
                    <div class="section">
                        <h3>Documentation:</h3>
                        <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
                        <p>Visit <a href="/redoc">/redoc</a> for alternative documentation</p>
                    </div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logging.error(f"Error serving web interface: {str(e)}")
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug
    )