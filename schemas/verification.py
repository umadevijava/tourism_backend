"""Schemas for fact verification and web source validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class SourceCredibility(str, Enum):
    """Source credibility levels."""
    VERY_HIGH = "very_high"  # Wikipedia, official .gov, academic
    HIGH = "high"  # News outlets, established publications
    MEDIUM = "medium"  # Blogs, forums with good reputation
    LOW = "low"  # Unknown sources
    

class SearchSource(BaseModel):
    """Represents a single web source."""
    title: str
    url: str
    snippet: str
    source_type: str = "general"  # "wikipedia", "map", "news", "official", etc.
    credibility: SourceCredibility = SourceCredibility.MEDIUM
    retrieved_at: datetime = datetime.now()
    

class LocationHierarchy(BaseModel):
    """Strict location hierarchy validation."""
    place: str
    district: Optional[str] = None
    state: str
    country: str = "India"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are populated."""
        return all([self.place, self.district, self.state, self.country])
    

class FactVerificationResult(BaseModel):
    """Result of verifying a specific fact."""
    fact: str
    is_verified: bool
    confidence: float  # 0.0 to 1.0
    sources: List[SearchSource]
    cross_checked: bool  # Whether checked against multiple sources
    verification_status: str  # "verified", "partially_verified", "unverified", "contradicted"
    notes: Optional[str] = None
    

class LocationVerificationResult(BaseModel):
    """Result of location hierarchy validation."""
    query: str
    location: Optional[LocationHierarchy] = None
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    sources: List[SearchSource]
    suggestions: List[str] = []  # Alternative matches if ambiguous
    osm_data: Optional[Dict[str, Any]] = None  # Raw OpenStreetMap data
    

class StructuredAnswer(BaseModel):
    """Standardized answer format with mandatory source attribution."""
    # For location queries
    place: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    
    # General answer
    answer: str
    
    # Sources and confidence
    sources: List[SearchSource]
    overall_confidence: str  # "High", "Medium", "Low"
    
    # Verification metadata
    verified: bool
    cross_checked: bool
    verification_method: str  # "web_search", "rag", "hybrid"
    requires_clarification: bool = False
    clarification_needed: Optional[str] = None
    
    # Logging
    query: str
    search_terms: List[str] = []
    retrieved_at: datetime = datetime.now()
    

class VerificationLog(BaseModel):
    """Complete verification log for debugging."""
    query: str
    query_type: str  # "location", "factual", "general"
    timestamp: datetime
    search_results_count: int
    sources_used: List[SearchSource]
    verification_steps: List[str]
    final_confidence: float
    final_answer: str
    cross_check_results: Optional[Dict[str, Any]] = None
