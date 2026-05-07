"""Query classifier and verification logging."""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from backend.schemas.verification import VerificationLog, StructuredAnswer, SearchSource
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of queries that require different verification strategies."""
    LOCATION = "location"  # "Where is Ponnur?", "What district is Delhi in?"
    FACTUAL = "factual"  # "What is the capital of India?"
    TEMPORAL = "temporal"  # "When did...", "What year...?"
    PERSON = "person"  # "Who is...", "Information about..."
    GENERAL = "general"  # Conversational, greeting, etc.


class QueryClassifier:
    """Classifies queries to determine verification strategy."""
    
    LOCATION_KEYWORDS = {
        "where", "location", "city", "district", "state", "country",
        "place", "situated", "located", "village", "town", "region",
        "address", "landmark", "temple", "monument", "building",
        "geographically", "map", "coordinates", "latitude", "longitude"
    }
    
    FACTUAL_KEYWORDS = {
        "what is", "what are", "explain", "define", "tell me about",
        "information about", "facts about", "describe", "details",
        "composed of", "made of", "consists of", "population of",
        "capital of", "currency of", "language of"
    }
    
    TEMPORAL_KEYWORDS = {
        "when", "year", "date", "time", "happened", "occurred",
        "was it", "took place", "established", "founded", "created",
        "started", "ended", "began", "concluded", "lasted"
    }
    
    PERSON_KEYWORDS = {
        "who is", "who was", "biography", "information about",
        "famous", "author", "scientist", "politician", "leader",
        "born", "died", "achievement", "known for", "invented"
    }
    
    @staticmethod
    def classify(query: str) -> QueryType:
        """
        Classify query type to determine verification strategy.
        
        Args:
            query: User's query
            
        Returns:
            QueryType indicating what kind of query this is
        """
        query_lower = query.lower()
        
        # Check location queries
        if any(keyword in query_lower for keyword in QueryClassifier.LOCATION_KEYWORDS):
            return QueryType.LOCATION
        
        # Check temporal queries
        if any(keyword in query_lower for keyword in QueryClassifier.TEMPORAL_KEYWORDS):
            return QueryType.TEMPORAL
        
        # Check person queries
        if any(keyword in query_lower for keyword in QueryClassifier.PERSON_KEYWORDS):
            return QueryType.PERSON
        
        # Check factual queries
        if any(keyword in query_lower for keyword in QueryClassifier.FACTUAL_KEYWORDS):
            return QueryType.FACTUAL
        
        # Default to general
        return QueryType.GENERAL


class VerificationLogger:
    """Logs verification process for debugging and analytics."""
    
    def __init__(self):
        """Initialize verification logger."""
        self.logs: List[VerificationLog] = []
        self.log_file = "verification_logs.jsonl"
    
    def log_verification(
        self,
        query: str,
        query_type: QueryType,
        search_results_count: int,
        sources: List[SearchSource],
        verification_steps: List[str],
        final_confidence: float,
        final_answer: str,
        cross_check_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a verification attempt.
        
        Args:
            query: User query
            query_type: Type of query
            search_results_count: Number of search results found
            sources: Sources used in verification
            verification_steps: Steps taken during verification
            final_confidence: Final confidence score
            final_answer: Generated answer
            cross_check_results: Results of cross-checking
        """
        log = VerificationLog(
            query=query,
            query_type=query_type.value,
            timestamp=datetime.now(),
            search_results_count=search_results_count,
            sources_used=sources,
            verification_steps=verification_steps,
            final_confidence=final_confidence,
            final_answer=final_answer[:200],  # Truncate for logging
            cross_check_results=cross_check_results,
        )
        
        self.logs.append(log)
        
        # Log to file
        try:
            with open(self.log_file, "a") as f:
                f.write(log.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to write verification log to file: {e}")
        
        # Log to console
        logger.info(
            f"Verification Log: query='{query[:50]}...' "
            f"type={query_type.value} sources={search_results_count} "
            f"confidence={final_confidence:.2f}"
        )
    
    def get_recent_logs(self, limit: int = 10) -> List[VerificationLog]:
        """
        Get recent verification logs.
        
        Args:
            limit: Number of logs to return
            
        Returns:
            List of recent verification logs
        """
        return self.logs[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get verification statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        if not self.logs:
            return {}
        
        total_queries = len(self.logs)
        avg_confidence = sum(log.final_confidence for log in self.logs) / total_queries
        
        query_types = {}
        for log in self.logs:
            query_types[log.query_type] = query_types.get(log.query_type, 0) + 1
        
        avg_sources = sum(log.search_results_count for log in self.logs) / total_queries
        
        return {
            "total_queries": total_queries,
            "avg_confidence": avg_confidence,
            "query_types": query_types,
            "avg_sources_per_query": avg_sources,
            "high_confidence_rate": sum(
                1 for log in self.logs if log.final_confidence >= 0.8
            ) / total_queries,
        }


# Global verification logger instance
_verification_logger = VerificationLogger()


def get_verification_logger() -> VerificationLogger:
    """Get the global verification logger instance."""
    return _verification_logger
