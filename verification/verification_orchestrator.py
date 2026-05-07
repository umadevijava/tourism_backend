"""Main verification orchestrator - coordinates all verification components."""

import asyncio
from typing import Optional, List, Dict, Any

from backend.schemas.verification import StructuredAnswer
from backend.verification.fact_verifier import FactVerifier
from backend.verification.location_validator import LocationValidator
from backend.verification.structured_output import StructuredOutputFormatter
from backend.verification.verification_logger import (
    QueryClassifier,
    QueryType,
    get_verification_logger,
)
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class VerificationOrchestrator:
    """
    Orchestrates the entire verification workflow.
    
    Workflow:
    1. Classify query type
    2. Extract intent and parameters
    3. Perform appropriate verification:
       - Location: Use LocationValidator
       - Factual: Use FactVerifier
       - General: Optional verification
    4. Format result in StructuredAnswer
    5. Log verification process
    """
    
    def __init__(self):
        """Initialize verification orchestrator."""
        self.fact_verifier = FactVerifier()
        self.location_validator = LocationValidator()
        self.query_classifier = QueryClassifier()
        self.logger = get_verification_logger()
        logger.info("VerificationOrchestrator initialized")
    
    async def verify_and_answer(
        self,
        query: str,
        additional_context: Optional[str] = None,
    ) -> StructuredAnswer:
        """
        Main entry point: Verify query and return structured answer.
        
        Args:
            query: User's query
            additional_context: Optional context (e.g., from chat history)
            
        Returns:
            StructuredAnswer with verified information or appropriate response
        """
        try:
            # Step 1: Classify query
            query_type = self.query_classifier.classify(query)
            logger.info(f"Query classified as: {query_type.value}")
            
            # Step 2: Route to appropriate verifier
            if query_type == QueryType.LOCATION:
                return await self._verify_location_query(query)
            elif query_type == QueryType.FACTUAL:
                return await self._verify_factual_query(query)
            elif query_type == QueryType.TEMPORAL:
                return await self._verify_temporal_query(query)
            elif query_type == QueryType.PERSON:
                return await self._verify_person_query(query)
            else:
                # General query - optional verification
                return await self._handle_general_query(query)
        
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return StructuredOutputFormatter.create_unverified_answer(
                query,
                attempted_search=True,
                error_message=str(e)
            )
    
    async def _verify_location_query(self, query: str) -> StructuredAnswer:
        """
        Verify location queries using LocationValidator.
        
        Examples:
        - "Where is Ponnur?"
        - "What district is Ponnur in?"
        - "Which state is Delhi in?"
        
        Args:
            query: Location query
            
        Returns:
            StructuredAnswer with verified location
        """
        logger.info(f"Verifying location query: {query}")
        
        # Extract place name from query
        # Simple extraction: most queries follow pattern "Where is [place]?"
        place_name = self._extract_place_name(query)
        
        if not place_name:
            return StructuredOutputFormatter.create_unverified_answer(
                query,
                error_message="Could not extract location from query"
            )
        
        # Step 1: Validate location using OpenStreetMap
        verification_steps = ["Extracted place name from query"]
        result = await self.location_validator.validate_location(place_name)
        verification_steps.append(f"Validated location via OSM: {place_name}")
        
        if not result.is_valid or not result.location:
            # Could not validate unique location
            if result.suggestions:
                # Return suggestions
                suggestions_text = "\n".join([f"  - {s}" for s in result.suggestions])
                answer_text = (
                    f"Multiple matches found for '{place_name}':\n{suggestions_text}\n\n"
                    "Please clarify which one you meant."
                )
            else:
                answer_text = (
                    f"Could not find location '{place_name}' in available sources. "
                    "Please check the spelling or provide more context."
                )
            
            answer = StructuredOutputFormatter.create_unverified_answer(
                query,
                error_message=answer_text
            )
            
            # Log
            self.logger.log_verification(
                query=query,
                query_type=QueryType.LOCATION,
                search_results_count=len(result.sources),
                sources=result.sources,
                verification_steps=verification_steps,
                final_confidence=result.confidence,
                final_answer=answer_text,
            )
            
            return answer
        
        # Step 2: Validate hierarchy consistency
        is_consistent, error = await self.location_validator.validate_hierarchy_consistency(
            result.location
        )
        verification_steps.append(f"Validated hierarchy consistency: {is_consistent}")
        
        # Step 3: Format structured answer
        answer = StructuredOutputFormatter.create_location_answer(
            place=result.location.place,
            district=result.location.district,
            state=result.location.state,
            country=result.location.country,
            sources=result.sources,
            additional_info=f"Coordinates: {result.location.latitude}, {result.location.longitude}" if result.location.latitude else None,
        )
        
        # Log verification
        self.logger.log_verification(
            query=query,
            query_type=QueryType.LOCATION,
            search_results_count=len(result.sources),
            sources=result.sources,
            verification_steps=verification_steps,
            final_confidence=result.confidence,
            final_answer=answer.answer,
        )
        
        return answer
    
    async def _verify_factual_query(self, query: str) -> StructuredAnswer:
        """
        Verify factual queries using FactVerifier.
        
        Args:
            query: Factual query
            
        Returns:
            StructuredAnswer with verified fact
        """
        logger.info(f"Verifying factual query: {query}")
        
        verification_steps = ["Initiated fact verification"]
        
        # Step 1: Verify fact
        fact_result = await self.fact_verifier.verify_fact(query)
        verification_steps.append(f"Verified fact status: {fact_result.verification_status}")
        
        # Step 2: Format answer
        answer = StructuredOutputFormatter.create_factual_answer(
            fact=query,
            sources=fact_result.sources,
            confidence=fact_result.confidence,
            verified=fact_result.is_verified,
            cross_checked=fact_result.cross_checked,
        )
        
        # Log
        self.logger.log_verification(
            query=query,
            query_type=QueryType.FACTUAL,
            search_results_count=len(fact_result.sources),
            sources=fact_result.sources,
            verification_steps=verification_steps,
            final_confidence=fact_result.confidence,
            final_answer=answer.answer,
            cross_check_results={"status": fact_result.verification_status},
        )
        
        return answer
    
    async def _verify_temporal_query(self, query: str) -> StructuredAnswer:
        """
        Verify temporal/historical queries.
        
        Args:
            query: Temporal query
            
        Returns:
            StructuredAnswer with verified information
        """
        logger.info(f"Verifying temporal query: {query}")
        
        # Similar to factual verification
        return await self._verify_factual_query(query)
    
    async def _verify_person_query(self, query: str) -> StructuredAnswer:
        """
        Verify person/biography queries.
        
        Args:
            query: Person query
            
        Returns:
            StructuredAnswer with verified information
        """
        logger.info(f"Verifying person query: {query}")
        
        # Similar to factual verification
        return await self._verify_factual_query(query)
    
    async def _handle_general_query(self, query: str) -> StructuredAnswer:
        """
        Handle general/conversational queries.
        
        Args:
            query: General query
            
        Returns:
            StructuredAnswer (may not be fully verified for conversational queries)
        """
        logger.info(f"Handling general query: {query}")
        
        verification_steps = ["General query - optional verification"]
        
        # For general queries, we optionally verify if it looks like it might be factual
        if any(keyword in query.lower() for keyword in ["information", "tell me", "about"]):
            fact_result = await self.fact_verifier.verify_fact(query)
            verification_steps.append(f"Optional fact verification: {fact_result.verification_status}")
            
            return StructuredOutputFormatter.create_factual_answer(
                fact=query,
                sources=fact_result.sources,
                confidence=fact_result.confidence,
                verified=fact_result.is_verified,
                cross_checked=fact_result.cross_checked,
            )
        
        # For purely conversational queries, just acknowledge
        answer = StructuredOutputFormatter.create_unverified_answer(
            query,
            attempted_search=False,
        )
        
        return answer
    
    @staticmethod
    def _extract_place_name(query: str) -> Optional[str]:
        """
        Extract place name from location query.
        
        Args:
            query: Query like "Where is [place]?"
            
        Returns:
            Extracted place name or None
        """
        query_lower = query.lower()
        
        # Pattern: "where is [place]?"
        if "where is" in query_lower:
            parts = query.split("where is ", 1)
            if len(parts) > 1:
                place = parts[1].strip().rstrip("?").strip()
                return place if place else None
        
        # Pattern: "what district is [place] in?"
        if "district" in query_lower and "in" in query_lower:
            # Extract between "is" and "in"
            if "is " in query_lower:
                parts = query_lower.split("is ", 1)
                if len(parts) > 1:
                    place_part = parts[1].split(" in", 1)[0].strip()
                    return place_part if place_part else None
        
        # Pattern: "location of [place]"
        if "location of" in query_lower:
            parts = query.split("location of ", 1)
            if len(parts) > 1:
                place = parts[1].strip().rstrip("?").strip()
                return place if place else None
        
        # Fallback: return whole query as place name
        return query.rstrip("?").strip()


# Global orchestrator instance
_verification_orchestrator: Optional[VerificationOrchestrator] = None


def get_verification_orchestrator() -> VerificationOrchestrator:
    """Get or create the global verification orchestrator."""
    global _verification_orchestrator
    if _verification_orchestrator is None:
        _verification_orchestrator = VerificationOrchestrator()
    return _verification_orchestrator
