#!/usr/bin/env python3
"""
Basic Usage Example for Personal Work Agent

This example shows how to use the agent programmatically.
"""

import asyncio
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import PersonalWorkAgent, Config


async def main():
    """Basic usage example"""
    print("🤖 Personal Work Agent - Basic Usage Example")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Create agent instance
    agent = PersonalWorkAgent(config)
    
    # Example 1: Train the agent with some work data
    print("\n1. Training the agent with work data...")
    
    work_data = """
    Project Alpha Meeting Notes - March 15, 2024
    
    Attendees: John Smith, Sarah Johnson, Mike Chen
    
    Key Discussion Points:
    - Reviewed Q1 performance metrics
    - Discussed new feature implementation for user dashboard
    - Identified bottlenecks in current workflow
    - Set deadline for prototype completion: March 30th
    
    Action Items:
    - John: Complete database schema design by March 20th
    - Sarah: Create wireframes for new dashboard by March 22nd
    - Mike: Set up CI/CD pipeline by March 25th
    
    Next meeting: March 22nd at 2 PM
    """
    
    metadata = {
        "type": "meeting_notes",
        "project": "Project Alpha",
        "date": "2024-03-15",
        "participants": ["John Smith", "Sarah Johnson", "Mike Chen"]
    }
    
    success = await agent.train_on_data(work_data, metadata)
    print(f"Training successful: {success}")
    
    # Example 2: Train with task data
    print("\n2. Adding task information...")
    
    task_data = """
    Task: Implement user authentication system
    Priority: High
    Assigned to: John Smith
    Due date: March 28, 2024
    
    Requirements:
    - OAuth 2.0 integration
    - Multi-factor authentication
    - Password reset functionality
    - Session management
    
    Technical notes:
    - Use JWT tokens for session management
    - Integrate with existing user database
    - Ensure GDPR compliance
    """
    
    task_metadata = {
        "type": "task",
        "priority": "High",
        "assignee": "John Smith",
        "due_date": "2024-03-28"
    }
    
    success = await agent.train_on_data(task_data, task_metadata)
    print(f"Task training successful: {success}")
    
    # Example 3: Query the agent
    print("\n3. Querying the agent...")
    
    questions = [
        "What are the action items for John Smith?",
        "When is the next meeting?",
        "What is the deadline for Project Alpha prototype?",
        "Who is working on the authentication system?",
        "What are the requirements for user authentication?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = await agent.query(question)
        print(f"A: {response.response}")
        print(f"Confidence: {response.confidence:.2f}")
        if response.sources:
            print(f"Sources: {len(response.sources)} references")
    
    # Example 4: Analyze some data
    print("\n4. Analyzing data...")
    
    sample_data = {
        "project_metrics": {
            "completed_tasks": 15,
            "pending_tasks": 8,
            "overdue_tasks": 2,
            "team_members": 5
        },
        "performance": {
            "velocity": 3.2,
            "quality_score": 0.89,
            "client_satisfaction": 4.2
        }
    }
    
    analysis = await agent.analyze_data(sample_data, "json")
    print(f"Analysis status: {analysis.get('status')}")
    if analysis.get('insights'):
        print("Insights:")
        for insight in analysis['insights']:
            print(f"  - {insight}")
    
    # Example 5: Get agent statistics
    print("\n5. Agent statistics...")
    
    stats = await agent.get_stats()
    print(f"Knowledge base size: {stats.get('knowledge_base_size', 0)} items")
    print(f"Agent status: {stats.get('status')}")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    # Note: Make sure you have set up your .env file with required API keys
    asyncio.run(main())