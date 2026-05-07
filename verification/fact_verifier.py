"""Multi-source fact verification engine."""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import json

from backend.schemas.verification import (
    FactVerificationResult,
    SearchSource,
    SourceCredibility,
)
from chatbot.bot.tools.google_search import GoogleSearchTool
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class FactVerifier:
    """
    Verifies facts using multiple web sources.
    
    Strategy:
    1. Search for fact on Google (multiple sources)
    2. Search on Wikipedia (for factual accuracy)
    3. Cross-check at least 2 sources
    4. Score credibility of sources
    5. Return verification result
    """
    
    def __init__(self):
        """Initialize fact verifier."""
        self.search_tool = GoogleSearchTool()
        self.min_sources_for_verification = 2
        self.cache: Dict[str, FactVerificationResult] = {}
        logger.info("FactVerifier initialized")
    
    async def verify_fact(
        self,
        fact: str,
        max_retries: int = 2,
    ) -> FactVerificationResult:
        """
        Verify a fact against multiple sources.
        
        Args:
            fact: The fact to verify (e.g., "Delhi is the capital of India")
            max_retries: Number of retries if verification fails
            
        Returns:
            FactVerificationResult with verification status and sources
        """
        # Check cache first
        cache_key = fact.lower().strip()
        if cache_key in self.cache:
            logger.info(f"Using cached verification for: {fact[:50]}...")
            return self.cache[cache_key]
        
        logger.info(f"Verifying fact: {fact[:50]}...")
        
        # Extract key terms from fact
        key_terms = self._extract_key_terms(fact)
        if not key_terms:
            return FactVerificationResult(
                fact=fact,
                is_verified=False,
                confidence=0.0,
                sources=[],
                cross_checked=False,
                verification_status="unverified",
                notes="Could not extract key terms from fact"
            )
        
        # Search for fact across multiple sources
        sources = await self._search_multiple_sources(key_terms)
        
        if not sources:
            # Retry with different search terms
            if max_retries > 0:
                logger.info(f"No sources found, retrying ({max_retries} retries left)...")
                # Refine search with broader terms
                broader_terms = [term for term in key_terms[:2]]  # Use first 2 terms
                sources = await self._search_multiple_sources(broader_terms)
        
        # Verify and cross-check
        result = await self._cross_check_sources(fact, sources)
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def _search_multiple_sources(
        self,
        key_terms: List[str],
    ) -> List[SearchSource]:
        """
        Search for key terms across multiple sources.
        
        Args:
            key_terms: List of search terms
            
        Returns:
            List of SearchSource objects from multiple searches
        """
        sources: List[SearchSource] = []
        
        # Search 1: Google Search
        for term in key_terms[:2]:  # Search top 2 terms
            try:
                logger.info(f"Searching Google for: {term}")
                results = await self.search_tool.search(term, num_results=5)
                
                if results.get("success"):
                    for result in results.get("results", []):
                        source = SearchSource(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            snippet=result.get("snippet", ""),
                            source_type="search",
                            credibility=self._assess_credibility(result.get("url", ""))
                        )
                        sources.append(source)
            except Exception as e:
                logger.error(f"Error searching Google for '{term}': {e}")
        
        # Search 2: Wikipedia (high credibility source)
        try:
            logger.info(f"Searching Wikipedia for: {key_terms[0]}")
            wiki_results = await self._search_wikipedia(key_terms[0])
            sources.extend(wiki_results)
        except Exception as e:
            logger.warning(f"Error searching Wikipedia: {e}")
        
        # Deduplicate sources by URL
        unique_sources = {source.url: source for source in sources}
        
        logger.info(f"Found {len(unique_sources)} unique sources")
        return list(unique_sources.values())
    
    async def _search_wikipedia(self, term: str) -> List[SearchSource]:
        """
        Search Wikipedia for high-credibility information.
        
        Args:
            term: Search term
            
        Returns:
            List of Wikipedia sources
        """
        try:
            import aiohttp
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "srsearch": term,
                "srprop": "snippet",
                "srlimit": 3,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sources = []
                        
                        for result in data.get("query", {}).get("search", []):
                            wiki_url = f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}"
                            source = SearchSource(
                                title=result.get("title", ""),
                                url=wiki_url,
                                snippet=result.get("snippet", ""),
                                source_type="wikipedia",
                                credibility=SourceCredibility.VERY_HIGH
                            )
                            sources.append(source)
                        
                        return sources
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
        
        return []
    
    async def _cross_check_sources(
        self,
        fact: str,
        sources: List[SearchSource],
    ) -> FactVerificationResult:
        """
        Cross-check fact against multiple sources.
        
        Args:
            fact: Fact to verify
            sources: List of sources to check
            
        Returns:
            Verification result
        """
        if not sources:
            return FactVerificationResult(
                fact=fact,
                is_verified=False,
                confidence=0.0,
                sources=[],
                cross_checked=False,
                verification_status="unverified",
                notes="No sources found for verification"
            )
        
        # Check if sources contain relevant information
        # This is a simple keyword matching approach
        # In production, use NLP for better matching
        fact_keywords = set(word.lower() for word in fact.split() if len(word) > 3)
        
        supporting_sources = []
        for source in sources:
            snippet_lower = source.snippet.lower()
            matching_keywords = sum(1 for keyword in fact_keywords if keyword in snippet_lower)
            
            # If snippet contains significant overlap with fact keywords
            if matching_keywords >= len(fact_keywords) * 0.5:
                supporting_sources.append(source)
        
        # Determine verification status
        is_verified = len(supporting_sources) >= self.min_sources_for_verification
        cross_checked = len(sources) >= 2
        
        if is_verified:
            verification_status = "verified"
            confidence = min(0.95, 0.7 + (len(supporting_sources) * 0.1))
        elif len(supporting_sources) > 0:
            verification_status = "partially_verified"
            confidence = 0.5 + (len(supporting_sources) * 0.2)
        else:
            verification_status = "unverified"
            confidence = 0.2
        
        return FactVerificationResult(
            fact=fact,
            is_verified=is_verified,
            confidence=confidence,
            sources=supporting_sources if supporting_sources else sources[:3],
            cross_checked=cross_checked,
            verification_status=verification_status,
            notes=f"Verified against {len(supporting_sources)} sources"
        )
    
    def _extract_key_terms(self, fact: str) -> List[str]:
        """
        Extract key search terms from a fact.
        
        Args:
            fact: Fact string
            
        Returns:
            List of search terms (sorted by importance)
        """
        # Simple extraction: take non-common words
        common_words = {
            "the", "is", "are", "was", "were", "a", "an", "and", "or",
            "in", "on", "at", "to", "for", "of", "with", "by",
            "what", "who", "when", "where", "why", "how"
        }
        
        words = fact.lower().split()
        key_terms = [
            word.strip(".,!?;:").replace('"', '')
            for word in words
            if word.lower() not in common_words and len(word) > 2
        ]
        
        return key_terms[:4]  # Top 4 terms
    
    def _assess_credibility(self, url: str) -> SourceCredibility:
        """
        Assess source credibility based on URL.
        
        Args:
            url: Source URL
            
        Returns:
            SourceCredibility level
        """
        url_lower = url.lower()
        
        # Very high credibility
        if any(domain in url_lower for domain in [
            "wikipedia.org", ".edu", ".gov", "academic", 
            "bbc.co.uk", "bbc.com", "reuters.com", "apnews.com"
        ]):
            return SourceCredibility.VERY_HIGH
        
        # High credibility
        if any(domain in url_lower for domain in [
            "nytimes.com", "theguardian.com", "cnn.com", "bbc",
            "britannica.com", "oxford", "cambridge"
        ]):
            return SourceCredibility.HIGH
        
        # Medium credibility
        if any(domain in url_lower for domain in [
            "medium.com", "quora.com", "reddit.com"
        ]):
            return SourceCredibility.MEDIUM
        
        # Default to medium
        return SourceCredibility.MEDIUM
