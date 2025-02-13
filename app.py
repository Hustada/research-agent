import os
import re
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from services.models import ModelFactory
from services.research.chains import ResearchChainManager
from services.search.search_manager import SearchManager
from services.search.serp_provider import SerpSearchProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

if not SERPAPI_API_KEY:
    raise ValueError("Missing required SERPAPI_API_KEY in .env file.")

# At least one AI provider must be configured
if not any([OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY]):
    raise ValueError("At least one AI provider API key must be set (OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY)")

# Map of available models
AVAILABLE_MODELS = {
    'provider': [],
    'api_keys': {}
}

if OPENAI_API_KEY:
    AVAILABLE_MODELS['provider'].append('openai')
    AVAILABLE_MODELS['api_keys']['openai'] = OPENAI_API_KEY

if GEMINI_API_KEY:
    AVAILABLE_MODELS['provider'].append('gemini')
    AVAILABLE_MODELS['api_keys']['gemini'] = GEMINI_API_KEY

if ANTHROPIC_API_KEY:
    AVAILABLE_MODELS['provider'].append('anthropic')
    AVAILABLE_MODELS['api_keys']['anthropic'] = ANTHROPIC_API_KEY

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html', available_models=AVAILABLE_MODELS['provider'])

def sanitize_input(input_text: str, max_length: int = 200) -> str:
    """Sanitize user input by removing potentially dangerous characters and limiting length."""
    if not input_text:
        return ""
    # Remove any non-alphanumeric characters except basic punctuation
    sanitized = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', input_text)
    # Limit length
    return sanitized[:max_length]

# Research depth prompts
RESEARCH_PROMPTS = {
    'brief': lambda topic: (
        f"Research '{topic}' and provide a concise 2-3 sentence overview that explains "
        f"ONLY the most essential facts someone needs to know. Focus on the present "
        f"state and most critical aspects."
    ),
    'detailed': lambda topic: (
        f"Research '{topic}' and provide a structured analysis with these specific sections:\n"
        f"1. Brief Definition/Overview (2-3 sentences)\n"
        f"2. Historical Context: Key developments and milestones\n"
        f"3. Current State: Major components and recent developments\n"
        f"4. Future Implications: Upcoming trends and potential impacts\n"
        f"Use specific examples and data points where possible."
    ),
    'comprehensive': lambda topic: (
        f"Perform an exhaustive analysis of '{topic}' with these required components:\n"
        f"1. Executive Summary (3-4 sentences)\n"
        f"2. Historical Evolution: Detailed timeline of major developments\n"
        f"3. Technical Deep-Dive: Core concepts, mechanisms, and relationships\n"
        f"4. Current Landscape: Key players, technologies, and methodologies\n"
        f"5. Challenges & Controversies: Major obstacles and debates\n"
        f"6. Future Outlook: Emerging trends, predictions, and potential breakthroughs\n"
        f"7. Expert Insights: Include specific quotes or findings from leading authorities\n"
        f"Use multiple sources and provide specific examples, statistics, and citations."
    )
}

@app.route('/research', methods=['POST'])
def research():
    """Handle research requests with comprehensive error handling."""
    try:
        # Validate and sanitize input
        data = request.get_json()
        topic = sanitize_input(data.get('topic', ''))
        depth = data.get('depth', 'brief').lower()

        # Validate inputs
        if not topic:
            return jsonify({
                'error': 'Topic is required and must contain valid characters'
            }), 400

        if depth not in RESEARCH_PROMPTS:
            return jsonify({
                'error': f"Invalid depth. Choose from: {', '.join(RESEARCH_PROMPTS.keys())}"
            }), 400

        # Log research attempt
        logger.info(f"Research request: topic={topic}, depth={depth}")

        # Get selected model provider (default to first available)
        model_provider = data.get('model', AVAILABLE_MODELS['provider'][0])
        
        if model_provider not in AVAILABLE_MODELS['provider']:
            return jsonify({
                'error': f"Invalid model provider. Choose from: {', '.join(AVAILABLE_MODELS['provider'])}"
            }), 400
            
        # Initialize components
        llm = ModelFactory.create_model(
            provider=model_provider,
            api_key=AVAILABLE_MODELS['api_keys'][model_provider],
            temperature=0.7,
            max_tokens=1500
        )
        research_manager = ResearchChainManager(llm)
        
        # Initialize search
        search_provider = SerpSearchProvider(api_key=SERPAPI_API_KEY)
        search_manager = SearchManager(search_provider=search_provider)
        
        # Get search results
        search_results = search_manager.search(topic, num_results=4)
        
        # Get research prompt
        research_prompt = RESEARCH_PROMPTS[depth](topic)
        
        # Process research
        research_output = research_manager.process_research(
            query=topic,
            prompt=research_prompt,
            sources=search_results,
            depth=depth
        )
        
        # Format response
        response = {
            'result': research_output['result'],
            'duration': research_output['duration'],
            'depth': research_output['depth'],
            'sources': [
                {
                    'title': s.title,
                    'url': s.url,
                    'snippet': s.snippet
                } for s in search_results[:5]  # Top 5 sources
            ]
        }
        
        # Log successful research
        logger.info(f"Research completed successfully for topic: {topic}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Research error: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred during research. Please try again.',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)
