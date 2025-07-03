"""
Query Parser - Natural Language Understanding for Spatial Queries
================================================================

This module handles parsing and understanding natural language spatial queries,
extracting key components like:
- Spatial intent (find, analyze, map, etc.)
- Geographic location and boundaries  
- Spatial constraints and filters
- Analysis requirements

Example Queries:
- "Find best places to build schools in flood-free zones near highways in Kerala"
- "Map flood risk zones in Kerala using elevation and rainfall data"
- "Identify optimal solar farm locations in Gujarat considering slope and land use"
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SpatialIntent(Enum):
    """Types of spatial analysis intentions"""
    FIND_LOCATIONS = "find_locations"
    MAP_ANALYSIS = "map_analysis" 
    SUITABILITY_ANALYSIS = "suitability_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PROXIMITY_ANALYSIS = "proximity_analysis"
    OVERLAY_ANALYSIS = "overlay_analysis"
    BUFFER_ANALYSIS = "buffer_analysis"
    ROUTE_ANALYSIS = "route_analysis"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed spatial query"""
    
    # Core Intent
    intent: SpatialIntent
    action_verbs: List[str]
    
    # Geographic Context
    location: Optional[str] = None
    admin_level: Optional[str] = None  # state, district, city, etc.
    coordinates: Optional[Tuple[float, float, float, float]] = None  # bbox
    
    # Analysis Requirements
    target_features: List[str] = None  # what to find/analyze
    constraints: List[str] = None      # restrictions/filters
    criteria: List[str] = None         # evaluation criteria
    
    # Data Requirements
    required_datasets: List[str] = None
    analysis_type: Optional[str] = None
    
    # Spatial Parameters
    buffer_distance: Optional[str] = None
    proximity_features: List[str] = None
    
    # Output Requirements
    output_format: str = "map"
    visualization_type: str = "interactive"
    
    # Confidence and Metadata
    confidence_score: float = 0.0
    raw_query: str = ""
    extracted_entities: Dict = None

    def __post_init__(self):
        if self.target_features is None:
            self.target_features = []
        if self.constraints is None:
            self.constraints = []
        if self.criteria is None:
            self.criteria = []
        if self.required_datasets is None:
            self.required_datasets = []
        if self.proximity_features is None:
            self.proximity_features = []
        if self.extracted_entities is None:
            self.extracted_entities = {}


class QueryParser:
    """
    Natural Language Query Parser for Spatial Analysis
    
    This class uses pattern matching, keyword extraction, and LLM assistance
    to parse natural language queries into structured spatial analysis requirements.
    """
    
    def __init__(self):
        """Initialize the query parser with pattern dictionaries"""
        
        # Intent patterns
        self.intent_patterns = {
            SpatialIntent.FIND_LOCATIONS: [
                r'\b(find|locate|identify|search for|discover)\b.*\b(best|optimal|suitable|good)\b.*\b(place|location|site|area|zone)\b',
                r'\b(where|which areas?)\b.*\b(best|suitable|optimal)\b',
                r'\b(site selection|location selection)\b'
            ],
            SpatialIntent.SUITABILITY_ANALYSIS: [
                r'\b(suitability|suitable|optimal|best)\b.*\b(analysis|assessment|evaluation)\b',
                r'\b(rank|prioritize|evaluate)\b.*\b(sites?|locations?|areas?)\b',
                r'\b(multi.?criteria|mcda)\b'
            ],
            SpatialIntent.MAP_ANALYSIS: [
                r'\b(map|mapping|visualiz|show|display)\b',
                r'\b(create|generate|produce)\b.*\bmaps?\b'
            ],
            SpatialIntent.RISK_ASSESSMENT: [
                r'\b(risk|hazard|vulnerability|danger)\b.*\b(assessment|analysis|mapping)\b',
                r'\b(flood|fire|earthquake|disaster)\b.*\b(risk|zone|prone)\b'
            ],
            SpatialIntent.PROXIMITY_ANALYSIS: [
                r'\bnear\b|\bclose to\b|\bwithin\b.*\b(distance|meters?|km|kilometers?)\b',
                r'\b(proximity|distance|accessibility)\b.*\b(analysis|assessment)\b'
            ],
            SpatialIntent.BUFFER_ANALYSIS: [
                r'\bbuffer\b|\bwithin\b.*\b\d+\s*(m|km|meters?|kilometers?)\b',
                r'\b\d+\s*(m|km|meters?|kilometers?)\b.*\b(radius|distance|buffer)\b'
            ]
        }
        
        # Geographic entities
        self.location_patterns = {
            'state': r'\b(Kerala|Gujarat|Maharashtra|Karnataka|Tamil Nadu|Rajasthan|Uttar Pradesh|Madhya Pradesh|Bihar|West Bengal|Andhra Pradesh|Telangana|Odisha|Punjab|Haryana|Himachal Pradesh|Uttarakhand|Jharkhand|Chhattisgarh|Goa|Manipur|Meghalaya|Tripura|Mizoram|Arunachal Pradesh|Nagaland|Sikkim|Assam|Jammu and Kashmir|Ladakh)\b',
            'city': r'\b(Mumbai|Delhi|Bangalore|Hyderabad|Chennai|Kolkata|Ahmedabad|Pune|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Patna|Vadodara|Ghaziabad)\b',
            'district': r'\b\w+\s+district\b|\bdistrict\s+\w+\b'
        }
        
        # Feature and constraint patterns
        self.feature_patterns = {
            'schools': r'\b(school|educational institution|learning center)\b',
            'hospitals': r'\b(hospital|medical center|health facility|clinic)\b',
            'roads': r'\b(road|highway|street|path|route|transportation)\b',
            'rivers': r'\b(river|stream|water body|waterway)\b',
            'forests': r'\b(forest|woodland|tree cover|vegetation)\b',
            'agriculture': r'\b(agricultural|farming|crop|cultivation)\b',
            'urban': r'\b(urban|city|built.?up|developed)\b',
            'elevation': r'\b(elevation|height|altitude|topography|dem)\b',
            'slope': r'\b(slope|gradient|steepness|terrain)\b',
            'flood': r'\b(flood|flooding|inundation|waterlogging)\b',
            'solar': r'\b(solar|photovoltaic|renewable energy)\b'
        }
        
        # Distance extraction patterns
        self.distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)',
            r'within\s+(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)',
            r'(\d+(?:\.\d+)?)\s*(m|meter|meters|km|kilometer|kilometers)\s+(?:from|of|radius)'
        ]
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured components
        
        Args:
            query: Natural language spatial query
            
        Returns:
            ParsedQuery: Structured query representation
        """
        logger.info(f"Parsing query: {query}")
        
        query_lower = query.lower()
        
        # Initialize parsed query
        parsed = ParsedQuery(
            intent=self._extract_intent(query_lower),
            action_verbs=self._extract_action_verbs(query_lower),
            raw_query=query
        )
        
        # Extract geographic information
        parsed.location = self._extract_location(query)
        parsed.admin_level = self._extract_admin_level(query)
        
        # Extract features and constraints
        parsed.target_features = self._extract_features(query_lower)
        parsed.constraints = self._extract_constraints(query_lower)
        parsed.criteria = self._extract_criteria(query_lower)
        
        # Extract spatial parameters
        parsed.buffer_distance = self._extract_distance(query_lower)
        parsed.proximity_features = self._extract_proximity_features(query_lower)
        
        # Determine required datasets
        parsed.required_datasets = self._infer_datasets(parsed)
        
        # Calculate confidence score
        parsed.confidence_score = self._calculate_confidence(parsed)
        
        logger.info(f"Parsed intent: {parsed.intent}, confidence: {parsed.confidence_score:.2f}")
        
        return parsed
    
    def _extract_intent(self, query: str) -> SpatialIntent:
        """Extract the primary spatial analysis intent"""
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score
        
        # Return intent with highest score, default to FIND_LOCATIONS
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return SpatialIntent.FIND_LOCATIONS
    
    def _extract_action_verbs(self, query: str) -> List[str]:
        """Extract action verbs from the query"""
        action_verbs = []
        verb_patterns = [
            r'\b(find|locate|identify|search|discover|map|analyze|assess|evaluate|rank|prioritize|show|display|create|generate)\b'
        ]
        
        for pattern in verb_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            action_verbs.extend(matches)
        
        return list(set(action_verbs))  # Remove duplicates
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract geographic location from query"""
        
        for location_type, pattern in self.location_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_admin_level(self, query: str) -> Optional[str]:
        """Determine administrative level (state, district, city)"""
        
        for level, pattern in self.location_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return level
        
        return None
    
    def _extract_features(self, query: str) -> List[str]:
        """Extract target features to analyze"""
        
        features = []
        for feature, pattern in self.feature_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                features.append(feature)
        
        return features
    
    def _extract_constraints(self, query: str) -> List[str]:
        """Extract constraints and filters"""
        
        constraints = []
        
        # Common constraint patterns
        constraint_patterns = [
            r'avoid(?:ing)?\s+([^,\.]+)',
            r'(?:not|no|free from|excluding)\s+([^,\.]+)',
            r'without\s+([^,\.]+)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            constraints.extend([match.strip() for match in matches])
        
        return constraints
    
    def _extract_criteria(self, query: str) -> List[str]:
        """Extract evaluation criteria"""
        
        criteria = []
        
        criteria_patterns = [
            r'considering\s+([^,\.]+)',
            r'based on\s+([^,\.]+)',
            r'using\s+([^,\.]+)',
            r'with\s+([^,\.]+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            criteria.extend([match.strip() for match in matches])
        
        return criteria
    
    def _extract_distance(self, query: str) -> Optional[str]:
        """Extract distance/buffer specifications"""
        
        for pattern in self.distance_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    distance, unit = match.group(1), match.group(2)
                    return f"{distance} {unit}"
                else:
                    return match.group(0)
        
        return None
    
    def _extract_proximity_features(self, query: str) -> List[str]:
        """Extract features for proximity analysis"""
        
        proximity_features = []
        
        # Look for "near X" patterns
        near_pattern = r'near\s+([^,\.]+)'
        matches = re.findall(near_pattern, query, re.IGNORECASE)
        proximity_features.extend([match.strip() for match in matches])
        
        return proximity_features
    
    def _infer_datasets(self, parsed: ParsedQuery) -> List[str]:
        """Infer required datasets based on parsed components"""
        
        datasets = []
        
        # Administrative boundaries
        if parsed.location:
            datasets.append("administrative_boundaries")
        
        # Based on features mentioned
        feature_dataset_map = {
            'schools': ['poi_education', 'osm_amenities'],
            'hospitals': ['poi_healthcare', 'osm_amenities'],
            'roads': ['road_network', 'osm_roads'],
            'rivers': ['water_bodies', 'osm_waterways'],
            'forests': ['land_cover', 'forest_cover'],
            'elevation': ['dem', 'srtm'],
            'slope': ['dem', 'srtm'],
            'flood': ['flood_zones', 'dem', 'rainfall'],
            'solar': ['solar_irradiance', 'land_cover'],
            'agriculture': ['land_use', 'crop_data']
        }
        
        for feature in parsed.target_features:
            if feature in feature_dataset_map:
                datasets.extend(feature_dataset_map[feature])
        
        # Based on intent
        if parsed.intent == SpatialIntent.RISK_ASSESSMENT:
            datasets.extend(['dem', 'land_cover'])
        elif parsed.intent == SpatialIntent.SUITABILITY_ANALYSIS:
            datasets.extend(['land_use', 'infrastructure'])
        
        return list(set(datasets))  # Remove duplicates
    
    def _calculate_confidence(self, parsed: ParsedQuery) -> float:
        """Calculate confidence score for the parsed query"""
        
        score = 0.0
        max_score = 100.0
        
        # Intent confidence
        if parsed.intent:
            score += 20.0
        
        # Location confidence
        if parsed.location:
            score += 20.0
        
        # Features confidence
        if parsed.target_features:
            score += 15.0
        
        # Action verbs confidence
        if parsed.action_verbs:
            score += 10.0
        
        # Constraints confidence
        if parsed.constraints:
            score += 10.0
        
        # Criteria confidence
        if parsed.criteria:
            score += 10.0
        
        # Dataset inference confidence
        if parsed.required_datasets:
            score += 15.0
        
        return min(score / max_score, 1.0)
    
    def to_json(self, parsed: ParsedQuery) -> str:
        """Convert parsed query to JSON representation"""
        
        return json.dumps({
            'intent': parsed.intent.value,
            'action_verbs': parsed.action_verbs,
            'location': parsed.location,
            'admin_level': parsed.admin_level,
            'target_features': parsed.target_features,
            'constraints': parsed.constraints,
            'criteria': parsed.criteria,
            'required_datasets': parsed.required_datasets,
            'buffer_distance': parsed.buffer_distance,
            'proximity_features': parsed.proximity_features,
            'confidence_score': parsed.confidence_score,
            'raw_query': parsed.raw_query
        }, indent=2)


# Example usage and testing
if __name__ == "__main__":
    parser = QueryParser()
    
    # Test queries
    test_queries = [
        "Find best places to build schools in flood-free zones near highways in Kerala",
        "Map flood risk zones in Kerala using elevation and rainfall data",
        "Identify optimal solar farm locations in Gujarat considering slope and land use",
        "Find suitable hospital locations within 2km of residential areas in Bangalore"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        parsed = parser.parse(query)
        print(parser.to_json(parsed))
