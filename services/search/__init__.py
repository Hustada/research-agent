from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    """Data class for standardized search results across providers"""
    title: str
    url: str
    snippet: str
    domain: str
    published_date: Optional[datetime] = None
    domain_authority: Optional[float] = None
    relevance_score: Optional[float] = None

class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute search and return standardized results"""
        pass
    
    def get_domain_authority(self, domain: str) -> float:
        """Get domain authority score"""
        return 0.7
