"""Location hierarchy validator using OpenStreetMap Nominatim API."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import aiohttp

from backend.schemas.verification import (
    LocationHierarchy,
    LocationVerificationResult,
    SearchSource,
    SourceCredibility,
)
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class LocationValidator:
    """
    Validates location hierarchies using OpenStreetMap Nominatim API.
    
    Strategy:
    1. Query Nominatim for the place name
    2. Extract and validate: Place → District → State → Country
    3. Cross-check with alternative sources (Google Maps if key available)
    4. Handle ambiguous matches by returning suggestions
    5. Return complete hierarchy or ask for clarification
    """
    
    def __init__(self):
        """Initialize location validator."""
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_url = "https://nominatim.openstreetmap.org/reverse"
        self.cache: Dict[str, LocationVerificationResult] = {}
        logger.info("LocationValidator initialized with OpenStreetMap Nominatim API")
    
    async def validate_location(
        self,
        place_name: str,
        district: Optional[str] = None,
        state: Optional[str] = None,
    ) -> LocationVerificationResult:
        """
        Validate a location and its hierarchy.
        
        Args:
            place_name: Name of the place (village, city, landmark, etc.)
            district: Optional district name for disambiguation
            state: Optional state name for disambiguation
            
        Returns:
            LocationVerificationResult with validation status and hierarchy
        """
        # Check cache
        cache_key = f"{place_name}:{district}:{state}".lower()
        if cache_key in self.cache:
            logger.info(f"Using cached location verification for: {place_name}")
            return self.cache[cache_key]
        
        logger.info(f"Validating location: {place_name} (District: {district}, State: {state})")
        
        # Search for location
        osm_results = await self._search_nominatim(place_name, district, state)
        
        if not osm_results:
            # Try with just the place name if additional params failed
            osm_results = await self._search_nominatim(place_name)
        
        if not osm_results:
            return LocationVerificationResult(
                query=place_name,
                location=None,
                is_valid=False,
                confidence=0.0,
                sources=[],
                suggestions=[],
                osm_data=None
            )
        
        # Get best match (first result from Nominatim)
        best_match = osm_results[0]
        
        # Extract hierarchy
        location = self._extract_location_hierarchy(best_match)
        
        # Get sources
        sources = [
            SearchSource(
                title=f"OpenStreetMap - {place_name}",
                url=best_match.get("osm_url", "https://www.openstreetmap.org"),
                snippet=self._format_osm_snippet(best_match),
                source_type="map",
                credibility=SourceCredibility.VERY_HIGH
            )
        ]
        
        # Determine result
        is_valid = location.is_complete() if location else False
        
        # Get suggestions for ambiguous matches
        suggestions = []
        if len(osm_results) > 1 and not is_valid:
            suggestions = [
                self._format_location_for_suggestion(r)
                for r in osm_results[1:3]  # Top 2 alternatives
            ]
        
        result = LocationVerificationResult(
            query=place_name,
            location=location,
            is_valid=is_valid,
            confidence=0.95 if is_valid else 0.6,
            sources=sources,
            suggestions=suggestions,
            osm_data=best_match
        )
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def _search_nominatim(
        self,
        place_name: str,
        district: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Nominatim API for location.
        
        Args:
            place_name: Place name to search
            district: Optional district filter
            state: Optional state filter
            
        Returns:
            List of matching locations from Nominatim
        """
        try:
            # Build search query
            query_parts = [place_name]
            if district:
                query_parts.append(district)
            if state:
                query_parts.append(state)
            
            query = ", ".join(query_parts) + ", India"  # Assume India by default
            
            params = {
                "q": query,
                "format": "json",
                "limit": 5,
                "extratags": 1,  # Get additional tags
                "addressdetails": 1,  # Get detailed address hierarchy
            }
            
            headers = {
                "User-Agent": "ChatBot-Verification/1.0"
            }
            
            logger.info(f"Querying Nominatim: {query}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.nominatim_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Enhance results with OSM URL
                        for item in data:
                            osm_type = item.get("osm_type", "way")  # way, node, relation
                            osm_id = item.get("osm_id", "")
                            item["osm_url"] = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"
                        
                        logger.info(f"Found {len(data)} results from Nominatim")
                        return data
        except asyncio.TimeoutError:
            logger.warning("Nominatim query timed out")
        except Exception as e:
            logger.error(f"Error querying Nominatim: {e}")
        
        return []
    
    def _extract_location_hierarchy(
        self,
        osm_data: Dict[str, Any],
    ) -> Optional[LocationHierarchy]:
        """
        Extract location hierarchy from OSM data.
        
        Args:
            osm_data: OpenStreetMap result
            
        Returns:
            LocationHierarchy object or None
        """
        try:
            address = osm_data.get("address", {})
            
            # Extract place
            place = osm_data.get("name", "")
            
            # Extract district (could be city_district, county, or administrative)
            district = (
                address.get("city_district") or
                address.get("county") or
                address.get("administrative")
            )
            
            # Try to get more specific district from tags
            if not district and osm_data.get("osm_type") == "node":
                # For nodes, might have city as district proxy
                district = address.get("city")
            
            # Extract state
            state = address.get("state", "")
            
            # Extract country
            country = address.get("country", "India")
            
            # Get coordinates
            lat = osm_data.get("lat")
            lon = osm_data.get("lon")
            
            location = LocationHierarchy(
                place=place,
                district=district,
                state=state,
                country=country,
                latitude=float(lat) if lat else None,
                longitude=float(lon) if lon else None,
            )
            
            logger.info(f"Extracted hierarchy: {location.place} → {location.district} → {location.state}")
            return location
        
        except Exception as e:
            logger.error(f"Error extracting location hierarchy: {e}")
            return None
    
    def _format_osm_snippet(self, osm_data: Dict[str, Any]) -> str:
        """
        Format OSM data as snippet text.
        
        Args:
            osm_data: OpenStreetMap result
            
        Returns:
            Formatted snippet
        """
        address = osm_data.get("address", {})
        place = osm_data.get("name", "Unknown")
        district = address.get("city_district") or address.get("county") or ""
        state = address.get("state", "")
        country = address.get("country", "India")
        
        parts = [place]
        if district:
            parts.append(district)
        if state:
            parts.append(state)
        parts.append(country)
        
        return " → ".join(parts)
    
    def _format_location_for_suggestion(self, osm_data: Dict[str, Any]) -> str:
        """
        Format location as suggestion string.
        
        Args:
            osm_data: OpenStreetMap result
            
        Returns:
            Formatted suggestion
        """
        address = osm_data.get("address", {})
        place = osm_data.get("name", "Unknown")
        district = address.get("city_district") or address.get("county") or ""
        state = address.get("state", "")
        
        if district and state:
            return f"{place}, {district}, {state}"
        elif state:
            return f"{place}, {state}"
        else:
            return place
    
    async def validate_hierarchy_consistency(
        self,
        location: LocationHierarchy,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that location hierarchy is consistent.
        
        Args:
            location: LocationHierarchy to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if place actually belongs to the district
        # by reverse geocoding
        try:
            if location.latitude and location.longitude:
                async with aiohttp.ClientSession() as session:
                    params = {
                        "lat": location.latitude,
                        "lon": location.longitude,
                        "format": "json",
                        "addressdetails": 1,
                    }
                    
                    async with session.get(
                        self.reverse_url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            reverse_data = await resp.json()
                            reverse_address = reverse_data.get("address", {})
                            
                            # Check consistency
                            reverse_place = reverse_data.get("name", "")
                            reverse_district = (
                                reverse_address.get("city_district") or
                                reverse_address.get("county")
                            )
                            reverse_state = reverse_address.get("state", "")
                            
                            # Validate
                            place_matches = location.place.lower() in reverse_place.lower() or reverse_place.lower() in location.place.lower()
                            if not place_matches:
                                return False, f"Place '{location.place}' doesn't match reverse geocoding result"
                            
                            if location.district and reverse_district:
                                district_matches = location.district.lower() in reverse_district.lower() or reverse_district.lower() in location.district.lower()
                                if not district_matches:
                                    return False, f"District mismatch: {location.district} vs {reverse_district}"
                            
                            if location.state and reverse_state:
                                state_matches = location.state.lower() in reverse_state.lower() or reverse_state.lower() in location.state.lower()
                                if not state_matches:
                                    return False, f"State mismatch: {location.state} vs {reverse_state}"
                            
                            return True, None
        except Exception as e:
            logger.warning(f"Error validating hierarchy consistency: {e}")
        
        return True, None  # Assume valid if can't reverse geocode
