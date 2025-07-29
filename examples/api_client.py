#!/usr/bin/env python3
"""
API Client Example for Personal Work Agent

This example shows how to interact with the agent via HTTP API.
"""

import requests
import json
import time


class AgentClient:
    """Simple client for interacting with the Personal Work Agent API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check if the agent is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def query(self, question, context_limit=5):
        """Ask a question to the agent"""
        data = {
            "question": question,
            "context_limit": context_limit
        }
        response = self.session.post(f"{self.base_url}/query", json=data)
        return response.json()
    
    def train(self, content, metadata=None):
        """Train the agent with text content"""
        data = {
            "content": content,
            "metadata": metadata or {}
        }
        response = self.session.post(f"{self.base_url}/train", json=data)
        return response.json()
    
    def train_file(self, file_path, metadata=None):
        """Train the agent with a file"""
        with open(file_path, 'rb') as f:
            files = {"file": f}
            data = {"metadata": json.dumps(metadata or {})}
            response = self.session.post(f"{self.base_url}/train/file", files=files, data=data)
        return response.json()
    
    def analyze(self, data, data_type="auto"):
        """Analyze data with the agent"""
        payload = {
            "data": data,
            "data_type": data_type
        }
        response = self.session.post(f"{self.base_url}/analyze", json=payload)
        return response.json()
    
    def get_stats(self):
        """Get agent statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        return response.json()
    
    def get_recent_interactions(self, limit=10):
        """Get recent interactions"""
        response = self.session.get(f"{self.base_url}/memory/recent?limit={limit}")
        return response.json()
    
    def search_interactions(self, query, limit=5):
        """Search through interaction history"""
        response = self.session.get(f"{self.base_url}/memory/search?query={query}&limit={limit}")
        return response.json()
    
    def add_feedback(self, interaction_id, rating, feedback_text=""):
        """Add feedback for an interaction"""
        data = {
            "interaction_id": interaction_id,
            "rating": rating,
            "feedback_text": feedback_text
        }
        response = self.session.post(f"{self.base_url}/feedback", json=data)
        return response.json()


def main():
    """Example usage of the API client"""
    print("🌐 Personal Work Agent - API Client Example")
    print("=" * 50)
    
    # Initialize client
    client = AgentClient()
    
    # Check if agent is running
    print("1. Checking agent health...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ Agent is not running: {health['error']}")
        print("Please start the agent with: python main.py")
        return
    
    print(f"✅ Agent is healthy: {health.get('agent_name')}")
    
    # Train the agent with sample data
    print("\n2. Training the agent...")
    
    training_content = """
    Weekly Team Standup - March 18, 2024
    
    Team Updates:
    - Alice completed the user interface mockups
    - Bob fixed the payment gateway integration bug
    - Charlie started working on the mobile app API
    
    Blockers:
    - Waiting for design approval from client
    - Database migration needs to be scheduled
    
    Next Week Goals:
    - Complete user testing phase
    - Deploy staging environment
    - Finalize Q1 deliverables
    """
    
    metadata = {
        "type": "standup",
        "date": "2024-03-18",
        "team": "Development Team"
    }
    
    train_result = client.train(training_content, metadata)
    print(f"Training result: {train_result.get('message')}")
    
    # Wait a moment for processing
    time.sleep(1)
    
    # Query the agent
    print("\n3. Querying the agent...")
    
    questions = [
        "What did Alice complete?",
        "What are the current blockers?",
        "What are the goals for next week?",
        "Who is working on the mobile app?",
        "When was the standup meeting?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = client.query(question)
        
        if "response" in result:
            print(f"A: {result['response']}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
        else:
            print(f"Error: {result}")
    
    # Analyze some sample data
    print("\n4. Analyzing project data...")
    
    project_data = {
        "sprint_metrics": {
            "story_points_completed": 32,
            "story_points_planned": 40,
            "bugs_found": 5,
            "bugs_fixed": 8,
            "team_velocity": 28.5
        },
        "team_info": {
            "developers": 4,
            "qa_engineers": 2,
            "product_owner": 1
        }
    }
    
    analysis = client.analyze(project_data)
    print(f"Analysis status: {analysis.get('status')}")
    if analysis.get('insights'):
        print("Key insights:")
        for insight in analysis['insights']:
            print(f"  • {insight}")
    
    # Get agent statistics
    print("\n5. Agent statistics...")
    
    stats = client.get_stats()
    if "error" not in stats:
        print(f"Knowledge base size: {stats.get('knowledge_base_size', 0)}")
        print(f"Agent status: {stats.get('status')}")
        if 'memory_stats' in stats:
            memory = stats['memory_stats']
            print(f"Total interactions: {memory.get('total_interactions', 0)}")
            print(f"Average confidence: {memory.get('average_confidence', 0):.3f}")
    
    # Get recent interactions
    print("\n6. Recent interactions...")
    
    recent = client.get_recent_interactions(3)
    if "interactions" in recent:
        interactions = recent["interactions"]
        print(f"Found {len(interactions)} recent interactions:")
        for i, interaction in enumerate(interactions[:3]):
            print(f"  {i+1}. Q: {interaction['question'][:60]}...")
            print(f"     A: {interaction['response'][:80]}...")
            print(f"     Confidence: {interaction.get('confidence', 0):.2f}")
    
    print("\n✅ API client example completed!")


if __name__ == "__main__":
    main()