from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import SearchResult, SearchProvider
from .serp_provider import SerpSearchProvider

class SearchManager:
    """Manages search operations using a configured provider"""
    
    def __init__(self, search_provider: SearchProvider):
        """Initialize with a search provider
        
        Args:
            search_provider: The search provider to use
        """
        self.search_provider = search_provider
            
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Execute search using the configured provider
        
        Args:
            query: Search query string
            num_results: Number of results to return
        
        Returns:
            List of SearchResult objects
        """
        try:
            # Execute search
            results = await self.search_provider.search(query, num_results)
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            raise
                    
        # Sort results by relevance score
        sorted_results = sorted(
            all_results,
            key=lambda x: x.relevance_score or 0.0,
            reverse=True
        )
        
        return sorted_results
    
    def get_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
