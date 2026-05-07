"""Structured output formatter for verified answers."""

from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.schemas.verification import (
    StructuredAnswer,
    SearchSource,
)
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class StructuredOutputFormatter:
    """
    Formats verified answers in structured, source-attributed format.
    
    Ensures all answers include:
    - Factual content (verified or explicitly marked uncertain)
    - Source attribution (URLs and titles)
    - Confidence level
    - Verification metadata
    """
    
    @staticmethod
    def create_location_answer(
        place: str,
        district: Optional[str],
        state: str,
        country: str,
        sources: List[SearchSource],
        additional_info: Optional[str] = None,
    ) -> StructuredAnswer:
        """
        Create a structured answer for location queries.
        
        Args:
            place: Place name
            district: District name
            state: State name
            country: Country name
            sources: Verified sources
            additional_info: Optional additional information
            
        Returns:
            StructuredAnswer ready for output
        """
        answer_text = f"{place} is located in {district}" if district else f"{place} is located in {state}"
        if district:
            answer_text += f", {state}"
        answer_text += f", {country}."
        
        if additional_info:
            answer_text += f"\n\n{additional_info}"
        
        return StructuredAnswer(
            place=place,
            district=district,
            state=state,
            country=country,
            answer=answer_text,
            sources=sources,
            overall_confidence="High" if len(sources) >= 2 else "Medium",
            verified=True,
            cross_checked=len(sources) >= 2,
            verification_method="location_hierarchy_validation",
            query=place,
            search_terms=[place, district or "", state],
        )
    
    @staticmethod
    def create_factual_answer(
        fact: str,
        sources: List[SearchSource],
        confidence: float = 0.0,
        verified: bool = False,
        cross_checked: bool = False,
    ) -> StructuredAnswer:
        """
        Create a structured answer for factual queries.
        
        Args:
            fact: The verified fact
            sources: Supporting sources
            confidence: Confidence score (0.0-1.0)
            verified: Whether fact is verified
            cross_checked: Whether checked against multiple sources
            
        Returns:
            StructuredAnswer ready for output
        """
        # Determine confidence level
        if confidence >= 0.8:
            confidence_str = "High"
        elif confidence >= 0.5:
            confidence_str = "Medium"
        else:
            confidence_str = "Low"
        
        return StructuredAnswer(
            answer=fact,
            sources=sources,
            overall_confidence=confidence_str,
            verified=verified,
            cross_checked=cross_checked,
            verification_method="multi_source_verification",
            query=fact[:50],
            search_terms=[],
        )
    
    @staticmethod
    def create_unverified_answer(
        query: str,
        attempted_search: bool = True,
        error_message: Optional[str] = None,
    ) -> StructuredAnswer:
        """
        Create an answer when verification fails.
        
        Args:
            query: Original query
            attempted_search: Whether search was attempted
            error_message: Error details
            
        Returns:
            StructuredAnswer with uncertainty indication
        """
        answer = (
            f"I couldn't verify this information from reliable sources. "
            f"The query was: '{query}'. "
        )
        
        if attempted_search and error_message:
            answer += f"Error: {error_message}. "
        
        answer += "Please try rephrasing your question or provide more context."
        
        return StructuredAnswer(
            answer=answer,
            sources=[],
            overall_confidence="Low",
            verified=False,
            cross_checked=False,
            verification_method="unverified",
            query=query,
            requires_clarification=True,
            clarification_needed="Please rephrase or provide additional context.",
        )
    
    @staticmethod
    def format_for_output(
        structured_answer: StructuredAnswer,
        include_raw_sources: bool = True,
    ) -> str:
        """
        Format StructuredAnswer for user-facing output.
        
        Args:
            structured_answer: The structured answer
            include_raw_sources: Include raw source list
            
        Returns:
            Formatted string for output
        """
        output = structured_answer.answer
        output += "\n\n"
        
        # Add confidence indicator
        output += f"**Confidence:** {structured_answer.overall_confidence}\n"
        
        # Add verification status
        if structured_answer.verified:
            output += "**Status:** ✅ Verified\n"
        else:
            output += "**Status:** ⚠️ Unverified\n"
        
        # Add source attribution
        if structured_answer.sources:
            output += "\n**Sources:**\n"
            for i, source in enumerate(structured_answer.sources[:3], 1):  # Top 3 sources
                output += f"{i}. [{source.title}]({source.url})\n"
                if source.snippet:
                    output += f"   *{source.snippet[:100]}...*\n"
        else:
            output += "\n**Note:** No reliable sources found for verification.\n"
        
        # Add clarification if needed
        if structured_answer.requires_clarification and structured_answer.clarification_needed:
            output += f"\n**Please Note:** {structured_answer.clarification_needed}\n"
        
        return output
    
    @staticmethod
    def format_structured_json(
        structured_answer: StructuredAnswer,
    ) -> Dict[str, Any]:
        """
        Format StructuredAnswer as JSON for API responses.
        
        Args:
            structured_answer: The structured answer
            
        Returns:
            Dictionary representation for JSON
        """
        return {
            "answer": structured_answer.answer,
            "location": {
                "place": structured_answer.place,
                "district": structured_answer.district,
                "state": structured_answer.state,
                "country": structured_answer.country,
            } if structured_answer.place else None,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "credibility": source.credibility.value,
                }
                for source in structured_answer.sources
            ],
            "verification": {
                "verified": structured_answer.verified,
                "cross_checked": structured_answer.cross_checked,
                "confidence": structured_answer.overall_confidence,
                "method": structured_answer.verification_method,
            },
            "metadata": {
                "query": structured_answer.query,
                "retrieved_at": structured_answer.retrieved_at.isoformat(),
                "requires_clarification": structured_answer.requires_clarification,
            }
        }
