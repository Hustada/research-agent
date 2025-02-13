from typing import List
import logging
from services.search.serp_provider import SearchProvider, SearchResult

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SearchManager:
    """Manages search operations using a configured search provider."""
    
    def __init__(self, search_provider: SearchProvider):
        """Initialize with a search provider."""
        self.search_provider = search_provider
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a search using the configured provider.
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 5)
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Performing search for query: {query}")
            results = self.search_provider.search(query, num_results)
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
