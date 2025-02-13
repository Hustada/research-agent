import os
import re
import uuid
import logging
from typing import Union, Dict, Any
import openai
from dotenv import load_dotenv, dotenv_values
from flask import Flask, render_template, request, jsonify, session
from langchain.agents import AgentExecutor, load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path, verbose=True)

# Explicitly set environment variables from .env file
env_vars = dotenv_values(env_path)
for key, value in env_vars.items():
    if value:
        os.environ[key] = value

# Retrieve API keys with explicit error handling
def get_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        logging.error(f"Environment variable {var_name} is not set!")
    return value

OPENAI_API_KEY = get_env_var('OPENAI_API_KEY')
SERPAPI_API_KEY = get_env_var('SERPAPI_API_KEY')
GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')
# Using ANTHROPIC_API_KEY instead of CLAUDE_API_KEY
ANTHROPIC_API_KEY = get_env_var('ANTHROPIC_API_KEY')

# Validate API keys at startup
if not OPENAI_API_KEY or not SERPAPI_API_KEY:
    logging.error("CRITICAL: OpenAI or SerpAPI key is missing.")
    print("CRITICAL: OpenAI or SerpAPI key is missing.")
    print("Loaded environment variables:", list(env_vars.keys()))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='research_agent.log'
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

def sanitize_input(input_text: str, max_length: int = 200) -> str:
    """
    Sanitize user input by removing potentially dangerous characters
    and limiting length.
    """
    # Remove non-alphanumeric characters except spaces and basic punctuation
    sanitized = re.sub(r'[^\w\s.,!?-]', '', input_text)
    # Truncate to max length
    return sanitized[:max_length]

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Enhanced output parsing with more robust handling
        """
        try:
            # If output is substantial, return as final answer
            if llm_output and len(llm_output.strip()) > 50:
                return AgentFinish(
                    return_values={"output": llm_output},
                    log=llm_output
                )
            
            # Fallback parsing
            raise ValueError(f"Insufficient output: `{llm_output}`")
        
        except Exception as e:
            logger.error(f"Output parsing error: {e}")
            raise

def create_research_agent(provider: str = 'openai', topic: str = '') -> AgentExecutor:
    """
    Create a research agent with dynamic provider selection
    """
    # Validate provider
    available_providers = ['openai']
    if GEMINI_AVAILABLE:
        available_providers.append('gemini')
    if CLAUDE_AVAILABLE:
        available_providers.append('claude')
    
    if provider not in available_providers:
        provider = 'openai'
        logger.warning(f"Invalid provider. Defaulting to OpenAI.")

    # Select Language Model
    try:
        if provider == 'openai':
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required")
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=1500
            )
        elif provider == 'gemini':
            from langchain.chat_models import ChatGooglePalm
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key is required")
            llm = ChatGooglePalm(
                google_api_key=GEMINI_API_KEY,
                temperature=0.7
            )
        elif provider == 'claude':
            from langchain.chat_models import ChatAnthropic
            if not CLAUDE_API_KEY:
                raise ValueError("Claude API key is required")
            llm = ChatAnthropic(
                anthropic_api_key=CLAUDE_API_KEY,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        logger.error(f"Provider initialization error: {e}")
        raise

    # Define tools
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Prepare conversation memory with unique session ID
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # Enhanced depth prompts with explicit tool usage and clear instructions
    depth_prompts = {
        'brief': (
            "Use the SerpAPI tool to quickly research the provided topic. "
            "Provide a concise 2-paragraph overview focusing on the most critical aspects. "
            "Include key facts, current status, and a brief future outlook."
        ),
        'detailed': (
            "Conduct a comprehensive search using SerpAPI for the topic. "
            "Create a detailed 3-paragraph summary that includes: "
            "1) Historical context, 2) Current state and key developments, "
            "3) Future implications and potential challenges. "
            "Cite sources and provide specific, verifiable information."
        ),
        'comprehensive': (
            "Perform an in-depth analysis using SerpAPI. "
            "Develop a comprehensive report that covers: "
            "1) Comprehensive historical background, "
            "2) Detailed current landscape and major players, "
            "3) Emerging trends and future predictions, "
            "4) Critical challenges and potential solutions. "
            "Include statistical data, expert insights, and multiple perspectives. "
            "Ensure all information is current and sourced from reputable references."
        )
    }

    # Initialize agent with enhanced configuration
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        output_parser=CustomOutputParser(),
        max_iterations=5  # Prevent infinite loops
    )

    return agent

@app.route('/')
def index():
    """
    Render main page with dynamically available providers
    """
    providers = ['openai']
    if GEMINI_AVAILABLE:
        providers.append('gemini')
    if CLAUDE_AVAILABLE:
        providers.append('claude')
    return render_template('index.html', providers=providers)

@app.route('/research', methods=['POST'])
def research():
    """
    Handle research requests with comprehensive error handling
    """
    try:
        # Validate and sanitize input
        data = request.json
        topic = sanitize_input(data.get('topic', ''))
        depth = data.get('depth', 'brief')
        provider = data.get('provider', 'openai')

        # Validate inputs
        if not topic:
            return jsonify({
                'error': 'Topic is required and must contain valid characters'
            }), 400

        # Validate depth
        depth_prompts = {
            'brief': f"Research '{topic}' and provide a concise 2-3 sentence overview that explains ONLY the most essential facts someone needs to know. Focus on the present state and most critical aspects. Do NOT include history or future implications.",
            
            'detailed': f"Research '{topic}' and provide a structured analysis with these specific sections:\n" \
                    f"1. Brief Definition/Overview (2-3 sentences)\n" \
                    f"2. Historical Context: Key developments and milestones\n" \
                    f"3. Current State: Major components and recent developments\n" \
                    f"4. Future Implications: Upcoming trends and potential impacts\n" \
                    f"Use specific examples and data points where possible.",
            
            'comprehensive': f"Perform an exhaustive analysis of '{topic}' with these required components:\n" \
                    f"1. Executive Summary (3-4 sentences)\n" \
                    f"2. Historical Evolution: Detailed timeline of major developments\n" \
                    f"3. Technical Deep-Dive: Core concepts, mechanisms, and relationships\n" \
                    f"4. Current Landscape: Key players, technologies, and methodologies\n" \
                    f"5. Challenges & Controversies: Major obstacles and debates\n" \
                    f"6. Future Outlook: Emerging trends, predictions, and potential breakthroughs\n" \
                    f"7. Expert Insights: Include specific quotes or findings from leading authorities\n" \
                    f"Use multiple sources and provide specific examples, statistics, and citations where possible."
        }
        if depth not in depth_prompts:
            return jsonify({
                'error': f"Invalid depth. Choose from: {', '.join(depth_prompts.keys())}"
            }), 400

        # Log research attempt
        logger.info(f"Research request: topic={topic}, depth={depth}, provider={provider}")

        # Create and run agent
        agent = create_research_agent(provider, topic)
        summary = agent.run(depth_prompts[depth])
        
        # Log successful research
        logger.info(f"Research completed successfully for topic: {topic}")

        return jsonify({
            'summary': summary,
            'conversation_id': session.get('conversation_id')
        })
    
    except Exception as e:
        # Comprehensive error handling
        logger.error(f"Research error: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred during research. Please try again.',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
