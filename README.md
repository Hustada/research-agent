# AI Research Agent

A sleek, minimalist web application that uses AI to generate research summaries on any topic, powered by multiple AI providers and enhanced web search capabilities.

## Features

- 🤖 Multi-AI Provider Support
  - OpenAI
  - Google Gemini
  - Anthropic Claude

- 🔍 Flexible Research Options
  - Multiple research depth levels (brief, detailed, comprehensive)
  - Web search integration with source tracking
  - Conversational AI agent with memory
  - Structured output with interactive citations
  - Modern UI with Tailwind CSS
  - Clean typography and visual hierarchy
  - Responsive design for all devices

## Enhancement Roadmap

### Phase 1: Search Enhancement
- [ ] Multiple search provider integration
- [ ] Search result caching
- [ ] Domain-specific search filters
- [ ] Query optimization

### Phase 2: Source Integration
- [ ] URL tracking and citation
- [ ] Source reliability scoring
- [ ] Domain authority checking
- [ ] Reference section generation

### Phase 3: Output Enhancement
- [ ] Structured JSON responses
- [ ] Confidence scoring
- [ ] Fact verification
- [ ] Interactive source exploration

## Prerequisites

- Python 3.8+
- OpenAI API Key
- SerpAPI Key

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup

1. Create a `.env` file in the project root
2. Add your API keys:
   ```
   OPENAI_API_KEY='your_openai_key'
   SERPAPI_API_KEY='your_serpapi_key'
   ```

⚠️ **Important Security Notes**:
- Never commit your `.env` file to version control
- Keep your API keys confidential
- The `.gitignore` file will prevent accidental exposure

## Running the Application

```bash
flask run
```

Visit `http://localhost:5000` in your browser.

## Customization

- Adjust `temperature` in `app.py` to control AI creativity
- Modify research depth prompts in the `/research` route

## Potential Monetization

- Offer tiered research services
- Create specialized research packages
- Develop custom agent configurations for specific industries
