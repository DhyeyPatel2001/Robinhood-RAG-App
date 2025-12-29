"""
Test Phoenix telemetry independently.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from config import settings


def test_phoenix():
    """Test Phoenix tracing with an LLM call."""
    print("=" * 50)
    print("Testing Phoenix Telemetry")
    print("=" * 50)
    
    # 1. Import and launch Phoenix
    print("\n1. Launching Phoenix...")
    import phoenix as px
    session = px.launch_app()
    print(f"   Phoenix UI: {session.url}")
    
    # 2. Register OTEL tracer
    print("\n2. Registering OTEL tracer...")
    from phoenix.otel import register
    tracer_provider = register(project_name="test-project")
    print("   Tracer registered")
    
    # 3. Instrument OpenAI
    print("\n3. Instrumenting OpenAI client...")
    from openinference.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("   OpenAI instrumented")
    
    # 4. Make an LLM call AFTER instrumentation
    print("\n4. Making test LLM call...")
    from openai import OpenAI
    
    client = OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",  # Use chat model for simpler test
        messages=[
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ],
        temperature=0.1,
    )
    
    answer = response.choices[0].message.content
    print(f"   Response: {answer}")
    
    # 5. Verify trace
    print("\n5. Checking for traces...")
    import time
    time.sleep(2)  # Give Phoenix time to process
    
    # Check if there are traces in the project
    try:
        project = px.Client().get_project(project_name="test-project")
        print(f"   Project found: {project.name}")
        
        # Get spans
        spans = px.Client().query_spans(project_name="test-project")
        if spans is not None and len(spans) > 0:
            print(f"   ✅ Found {len(spans)} spans!")
            print(f"   Span types: {spans['name'].tolist()[:5]}")
        else:
            print("   ⚠️ No spans found yet (may take a moment)")
    except Exception as e:
        print(f"   Error querying spans: {e}")
    
    print("\n" + "=" * 50)
    print(f"Check Phoenix UI at: {session.url}")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    test_phoenix()
    print("\nKeeping Phoenix running. Press Ctrl+C to exit...")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")
