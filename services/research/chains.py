from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any
import logging
from time import time
from ..search import SearchResult

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ResearchChainManager:
    def __init__(self, llm: OpenAI):
        """Initialize the research chain manager with an OpenAI LLM."""
        self.llm = llm
        
        # Common source analysis prompt (used for all depths)
        self.source_analysis_prompt = PromptTemplate(
            input_variables=["sources"],
            template=(
                "Analyze the following search results and evaluate their credibility, key findings, "
                "and relevance. Format your analysis as a structured list with clear sections.\n\n"
                "Sources:\n{sources}"
            )
        )
        self.source_analysis_chain = LLMChain(llm=llm, prompt=self.source_analysis_prompt)
        
        # Brief synthesis prompt
        self.brief_synthesis_prompt = PromptTemplate(
            input_variables=["source_analysis", "query", "prompt"],
            template=(
                "{prompt}\n\n"
                "Based on the analysis below:\n{source_analysis}\n\n"
                "Provide a concise 2-3 sentence overview answering the query:\n{query}\n\n"
                "Structure your response in clear paragraphs. "
                "If necessary, include a novel example of the topic to illustrate your point. "
                "Include inline citations (e.g., [Source 1]) where applicable."
            )
        )
        self.brief_chain = LLMChain(llm=llm, prompt=self.brief_synthesis_prompt)
        
        # Detailed synthesis prompt
        self.detailed_synthesis_prompt = PromptTemplate(
            input_variables=["source_analysis", "query", "prompt"],
            template=(
                "{prompt}\n\n"
                "Based on the analysis below:\n{source_analysis}\n\n"
                "Provide a detailed answer for the query:\n{query}\n\n"
                "Structure your response with these sections:\n\n"
                "## Overview\n\n"
                "## Key Findings\n\n"
                "## Analysis of Contradictions\n\n"
                "Ensure each section is clearly written and use inline citations (e.g., [Source 2]). "
                "Include relevant examples where appropriate."
            )
        )
        self.detailed_chain = LLMChain(llm=llm, prompt=self.detailed_synthesis_prompt)
        
        # Comprehensive synthesis prompt
        self.comprehensive_synthesis_prompt = PromptTemplate(
            input_variables=["source_analysis", "query", "prompt"],
            template=(
                "{prompt}\n\n"
                "Based on the analysis below:\n{source_analysis}\n\n"
                "Provide a comprehensive research report for the query:\n{query}\n\n"
                "Your report should include the following sections with proper headers:\n\n"
                "## Executive Summary\n\n"
                "## Historical Context\n\n"
                "## Technical Analysis\n\n"
                "## Challenges & Future Outlook\n\n"
                "## Expert Insights\n\n"
                "Ensure that the report is organized in clear paragraphs. "
                "If needed, provide novel examples of the topic to clarify complex points. "
                "Reference all sources with inline citations (e.g., [Source 3])."
            )
        )
        self.comprehensive_chain = LLMChain(llm=llm, prompt=self.comprehensive_synthesis_prompt)
    
    def _format_sources(self, sources: List[SearchResult]) -> str:
        """Format search results for LLM input."""
        logger.info(f"\nFormatting {len(sources)} sources for analysis...")
        format_start = time()
        
        try:
            formatted = []
            for i, source in enumerate(sources, 1):
                formatted.append(
                    f"[Source {i}]\n"
                    f"Title: {source.title}\n"
                    f"URL: {source.url}\n"
                    f"Snippet: {source.snippet}\n"
                )
            
            formatted_text = "\n".join(formatted)
            format_duration = time() - format_start
            logger.info(f"Source formatting completed in {format_duration:.2f}s")
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Source formatting error: {str(e)}")
            raise
    
    def process_research(self, query: str, prompt: str, sources: List[SearchResult], depth: str = "detailed") -> Dict[str, Any]:
        """Process a research query using the appropriate chain for the specified depth."""
        start_time = time()
        logger.info(f"\n{'='*80}\nStarting Research Query: {query}\nDepth: {depth}\n{'='*80}")
        
        try:
            # Format sources for analysis
            formatted_sources = self._format_sources(sources)
            
            # Run source analysis
            source_analysis = self.source_analysis_chain.run({"sources": formatted_sources})
            
            # Choose synthesis chain based on depth
            if depth.lower() == "brief":
                synthesis_chain = self.brief_chain
            elif depth.lower() == "detailed":
                synthesis_chain = self.detailed_chain
            elif depth.lower() == "comprehensive":
                synthesis_chain = self.comprehensive_chain
            else:
                raise ValueError(f"Invalid research depth: {depth}")
            
            # Run synthesis
            synthesis = synthesis_chain.run({
                "source_analysis": source_analysis,
                "query": query,
                "prompt": prompt
            })
            
            duration = time() - start_time
            
            # Format response
            response = {
                "result": synthesis,
                "duration": duration,
                "depth": depth
            }
            
            logger.info(f"\n{'='*80}\nResearch Complete\n{'='*80}")
            return response
            
        except Exception as e:
            logger.error(f"Research processing failed: {str(e)}")
            raise
