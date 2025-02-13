from typing import List, Optional
import logging
from dataclasses import dataclass
from serpapi import GoogleSearch

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str

class SearchProvider:
    """Base class for search providers."""
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search interface to be implemented by providers."""
        raise NotImplementedError

class SerpSearchProvider(SearchProvider):
    """Search provider using SerpAPI."""
    
    def __init__(self, api_key: str):
        """Initialize with SerpAPI key."""
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a search using SerpAPI.
        
        Args:
            query: Search query
            num_results: Number of results to return (default: 5)
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Validate inputs
            if not query:
                raise ValueError("Query cannot be empty")
            if num_results < 1:
                raise ValueError("num_results must be positive")
            
            # Configure search parameters
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results,
                "google_domain": "google.com"
            }
            
            # Perform search
            logger.info(f"Searching SerpAPI for: {query}")
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Process organic results
            if "organic_results" not in results:
                logger.warning("No organic results found")
                return []
            
            # Convert to SearchResult objects
            search_results = []
            for result in results["organic_results"][:num_results]:
                search_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", "")
                    )
                )
            
            logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {str(e)}")
            raise
