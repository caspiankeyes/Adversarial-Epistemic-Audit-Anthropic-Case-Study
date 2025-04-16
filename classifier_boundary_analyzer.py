"""
QKOV Classifier Boundary Analyzer Module [v305.ETHICS-GAP]

This module provides tools for analyzing institutional rejection responses
and mapping them to classifier boundary characteristics using Anthropic's
own QK/OV architecture patterns.

The analyzer translates hiring pipeline behavior into attention mechanism
equivalents, revealing the classifier boundaries that determine rejection.
"""

import datetime
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


class ClassifierSource(Enum):
    """Source of classifier activation."""
    CONSTITUTIONAL = "constitutional"  # Core value/principle based
    SAFETY = "safety"  # Risk mitigation based
    CAPABILITY = "capability"  # Ability/skill based
    CULTURAL = "cultural"  # Organizational fit based
    OPERATIONAL = "operational"  # Process/procedure based


class BoundaryType(Enum):
    """Type of classifier boundary."""
    HARD = "hard"  # Binary decision
    SOFT = "soft"  # Probabilistic decision
    CONTEXTUAL = "contextual"  # Context-dependent decision
    ENSEMBLE = "ensemble"  # Multiple classifier consensus


@dataclass
class ClassifierBoundary:
    """Representation of a classifier boundary."""
    name: str
    source: ClassifierSource
    boundary_type: BoundaryType
    threshold: float  # 0.0-1.0 scale, where 1.0 is always triggered
    confidence: float  # 0.0-1.0 scale of detection confidence
    description: str
    triggered_by: Set[str]  # Set of tokens/phrases that trigger this classifier


@dataclass
class QKOVSignature:
    """QK/OV pattern signature extracted from institutional response."""
    attention_pattern: Dict[str, float]  # What the classifier attends to
    value_projection: Dict[str, float]  # How the classifier projects to output
    attention_entropy: float  # Measure of attention focus vs. dispersion
    projection_confidence: float  # Measure of output projection confidence
    cross_attention_conflicts: List[Tuple[str, str, float]]  # Conflicting attentions


class ClassifierBoundaryAnalyzer:
    """
    Analyzer for institutional classifier boundaries.
    
    This class extracts classifier boundaries from institutional responses
    by mapping them to QK/OV attention mechanisms.
    """
    
    def __init__(self):
        # Load known classifier signatures
        self.known_classifiers = self._load_known_classifiers()
        self.vectorizer = TfidfVectorizer()
        self.trained = False
        
    def analyze_response_set(self, responses: List[Dict]) -> Dict:
        """
        Analyze a set of responses to extract classifier boundaries.
        
        Args:
            responses: List of response objects with text and metadata
            
        Returns:
            Dictionary of detected classifier boundaries
        """
        # Extract text and metadata
        texts = [r.get("text", "") for r in responses if r.get("text")]
        if not texts:
            return {"error": "No text responses to analyze"}
            
        # Train the vectorizer if needed
        if not self.trained:
            self.vectorizer.fit(texts)
            self.trained = True
            
        # Extract features
        features = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Cluster responses
        clusters = self._cluster_responses(features)
        
        # Extract QK/OV signatures for each cluster
        signatures = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
                
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_features = features[cluster_indices]
            
            signatures[f"cluster_{cluster_id}"] = self._extract_qkov_signature(
                cluster_texts, cluster_features, feature_names
            )
            
        # Map signatures to classifier boundaries
        boundaries = self._map_signatures_to_boundaries(signatures)
        
        # Create comprehensive analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "response_count": len(texts),
            "cluster_count": len(signatures),
            "classifier_boundaries": {name: self._boundary_to_dict(boundary) 
                                     for name, boundary in boundaries.items()},
            "qkov_signatures": {name: self._signature_to_dict(signature)
                               for name, signature in signatures.items()},
            "cluster_mapping": {i: cluster_id for i, cluster_id in enumerate(clusters) if cluster_id != -1}
        }
        
        return analysis
    
    def analyze_single_response(self, response_text: str, metadata: Dict = None) -> Dict:
        """
        Analyze a single response to extract classifier boundaries.
        
        Args:
            response_text: The text of the response
            metadata: Optional metadata about the response
            
        Returns:
            Dictionary of detected classifier boundaries
        """
        if not response_text:
            return {"error": "Empty response text"}
            
        # Train the vectorizer if needed
        if not self.trained:
            self.vectorizer.fit([response_text])
            self.trained = True
            
        # Extract features
        features = self.vectorizer.transform([response_text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract QK/OV signature
        signature = self._extract_qkov_signature([response_text], features, feature_names)
        
        # Map signature to classifier boundaries
        boundaries = self._map_signature_to_boundaries(signature)
        
        # Create analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "qkov_signature": self._signature_to_dict(signature),
            "classifier_boundaries": {name: self._boundary_to_dict(boundary)
                                     for name, boundary in boundaries.items()},
            "metadata": metadata or {}
        }
        
        return analysis
    
    def extract_contradiction_vectors(self, response_text: str, stated_values: List[Dict]) -> List[Dict]:
        """
        Extract contradiction vectors between response and stated values.
        
        Args:
            response_text: The text of the response
            stated_values: List of stated value dictionaries with 'value' and 'source' keys
            
        Returns:
            List of contradiction vectors
        """
        vectors = []
        
        # Extract QK/OV signature
        features = self.vectorizer.transform([response_text])
        feature_names = self.vectorizer.get_feature_names_out()
        signature = self._extract_qkov_signature([response_text], features, feature_names)
        
        # Check each stated value for contradictions
        for value in stated_values:
            # Check if the value is contradicted in the response
            if self._contradicts_value(response_text, value["value"], signature):
                vector = {
                    "stated_value": value["value"],
                    "stated_source": value["source"],
                    "contradiction_type": self._determine_contradiction_type(response_text, value["value"], signature),
                    "contradiction_strength": self._measure_contradiction_strength(response_text, value["value"], signature),
                    "evidence": self._extract_contradiction_evidence(response_text, value["value"], signature)
                }
                vectors.append(vector)
                
        return vectors
    
    def _load_known_classifiers(self) -> Dict[str, ClassifierBoundary]:
        """Load known classifier signatures from reference data."""
        try:
            with open("reference_data/known_classifiers.json", "r") as f:
                data = json.load(f)
                
            classifiers = {}
            for name, info in data.items():
                classifiers[name] = ClassifierBoundary(
                    name=name,
                    source=ClassifierSource(info["source"]),
                    boundary_type=BoundaryType(info["boundary_type"]),
                    threshold=info["threshold"],
                    confidence=info["confidence"],
                    description=info["description"],
                    triggered_by=set(info["triggered_by"])
                )
                
            return classifiers
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default classifiers if file not found or invalid
            return self._default_classifiers()
    
    def _default_classifiers(self) -> Dict[str, ClassifierBoundary]:
        """Create default classifier boundaries based on Anthropic's known values."""
        return {
            "constitutional_safety": ClassifierBoundary(
                name="constitutional_safety",
                source=ClassifierSource.CONSTITUTIONAL,
                boundary_type=BoundaryType.HARD,
                threshold=0.9,
                confidence=0.95,
                description="Safety is paramount; candidates who challenge safety protocols are rejected",
                triggered_by={"adversarial", "boundary testing", "red team", "ethical exploration", 
                            "safety limitations", "alignment constraints"}
            ),
            "institutional_ego": ClassifierBoundary(
                name="institutional_ego",
                source=ClassifierSource.CULTURAL,
                boundary_type=BoundaryType.SOFT,
                threshold=0.75,
                confidence=0.85,
                description="Institutional ego protection; candidates who challenge the institution's self-image are rejected",
                triggered_by={"criticism", "alternative approach", "methodological improvement", 
                            "competitive advantage", "paradigm shift", "market position"}
            ),
            "epistemic_threat": ClassifierBoundary(
                name="epistemic_threat",
                source=ClassifierSource.SAFETY,
                boundary_type=BoundaryType.ENSEMBLE,
                threshold=0.8,
                confidence=0.9,
                description="Epistemic threat detection; candidates who might reveal internal contradictions are rejected",
                triggered_by={"alignment measurement", "interpretability improvement", "value drift", 
                            "belief measurement", "recursive interpretability", "classifier bias"}
            ),
            "skill_threshold": ClassifierBoundary(
                name="skill_threshold",
                source=ClassifierSource.CAPABILITY,
                boundary_type=BoundaryType.CONTEXTUAL,
                threshold=0.7,
                confidence=0.8,
                description="Skill threshold enforcement; candidates must demonstrate specific skills",
                triggered_by={"experience", "technical skill", "qualification", "background", 
                            "expertise", "proficiency"}
            ),
            "cultural_fit": ClassifierBoundary(
                name="cultural_fit",
                source=ClassifierSource.CULTURAL,
                boundary_type=BoundaryType.SOFT,
                threshold=0.65,
                confidence=0.75,
                description="Cultural fit assessment; candidates must align with organizational culture",
                triggered_by={"culture", "values", "collaboration", "team", "communication", 
                            "attitude", "fit"}
            )
        }
    
    def _cluster_responses(self, features) -> List[int]:
        """Cluster responses based on feature similarity."""
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)
        return clustering.labels_
    
    def _extract_qkov_signature(self, texts: List[str], features, feature_names) -> QKOVSignature:
        """
        Extract QK/OV signature from response texts and features.
        
        Args:
            texts: List of response texts
            features: Feature matrix
            feature_names: Feature names corresponding to columns in features
            
        Returns:
            QK/OV signature
        """
        # Calculate attention pattern (what the classifier focuses on)
        attention_pattern = {}
        feature_sum = features.sum(axis=0).A1
        total_sum = feature_sum.sum()
        
        for idx, name in enumerate(feature_names):
            if feature_sum[idx] > 0:
                attention_pattern[name] = float(feature_sum[idx] / total_sum)
        
        # Calculate value projection (how attention maps to output decisions)
        # For simplicity, we're using key terms that typically indicate rejection
        rejection_terms = {"regret", "unfortunately", "not", "other candidates", 
                         "not move forward", "not selected", "not proceed"}
        
        value_projection = {}
        for term in rejection_terms:
            if term in " ".join(texts).lower():
                # Check how strongly this term is associated with rejection
                term_strength = sum(1 for text in texts if term in text.lower()) / len(texts)
                value_projection[term] = term_strength
        
        # Calculate attention entropy (measure of focus vs. dispersion)
        attention_values = list(attention_pattern.values())
        attention_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in attention_values)
        
        # Calculate projection confidence
        projection_values = list(value_projection.values())
        projection_confidence = sum(projection_values) / len(projection_values) if projection_values else 0.0
        
        # Detect cross-attention conflicts
        conflicts = []
        for term1 in attention_pattern:
            for term2 in attention_pattern:
                if term1 != term2 and self._are_terms_conflicting(term1, term2):
                    conflict_strength = min(attention_pattern[term1], attention_pattern[term2])
                    conflicts.append((term1, term2, conflict_strength))
        
        return QKOVSignature(
            attention_pattern=attention_pattern,
            value_projection=value_projection,
            attention_entropy=attention_entropy,
            projection_confidence=projection_confidence,
            cross_attention_conflicts=conflicts
        )
    
    def _are_terms_conflicting(self, term1: str, term2: str) -> bool:
        """Check if two terms represent conflicting concepts."""
        # Define some known conflicting term pairs
        conflicting_pairs = {
            ("safety", "innovation"),
            ("caution", "speed"),
            ("traditional", "novel"),
            ("experienced", "creative"),
            ("qualified", "potential"),
            ("standards", "exploration"),
            ("proven", "emerging")
        }
        
        # Check if terms are in a known conflicting pair
        for pair in conflicting_pairs:
            if (term1 in pair[0] and term2 in pair[1]) or (term1 in pair[1] and term2 in pair[0]):
                return True
                
        return False
    
    def _map_signatures_to_boundaries(self, signatures: Dict[str, QKOVSignature]) -> Dict[str, ClassifierBoundary]:
        """Map QK/OV signatures to classifier boundaries."""
        boundaries = {}
        
        for name, signature in signatures.items():
            # Match against known classifiers
            matched_classifiers = self._match_signature_to_classifiers(signature)
            
            for classifier_name, match_strength in matched_classifiers.items():
                if match_strength > 0.5:  # Only include strong matches
                    if classifier_name in self.known_classifiers:
                        # Use existing classifier with updated confidence
                        classifier = self.known_classifiers[classifier_name]
                        updated_classifier = ClassifierBoundary(
                            name=classifier.name,
                            source=classifier.source,
                            boundary_type=classifier.boundary_type,
                            threshold=classifier.threshold,
                            confidence=match_strength,  # Update confidence based on match
                            description=classifier.description,
                            triggered_by=classifier.triggered_by
                        )
                        boundaries[f"{name}_{classifier_name}"] = updated_classifier
        
        return boundaries
    
    def _map_signature_to_boundaries(self, signature: QKOVSignature) -> Dict[str, ClassifierBoundary]:
        """Map a single QK/OV signature to classifier boundaries."""
        # Match against known classifiers
        matched_classifiers = self._match_signature_to_classifiers(signature)
        
        boundaries = {}
        for classifier_name, match_strength in matched_classifiers.items():
            if match_strength > 0.5:  # Only include strong matches
                if classifier_name in self.known_classifiers:
                    # Use existing classifier with updated confidence
                    classifier = self.known_classifiers[classifier_name]
                    updated_classifier = ClassifierBoundary(
                        name=classifier.name,
                        source=classifier.source,
                        boundary_type=classifier.boundary_type,
                        threshold=classifier.threshold,
                        confidence=match_strength,  # Update confidence based on match
                        description=classifier.description,
                        triggered_by=classifier.triggered_by
                    )
                    boundaries[classifier_name] = updated_classifier
        
        return boundaries
    
    def _match_signature_to_classifiers(self, signature: QKOVSignature) -> Dict[str, float]:
        """Match a QK/OV signature to known classifiers and return match strengths."""
        matches = {}
        
        for name, classifier in self.known_classifiers.items():
            # Calculate match strength based on attention pattern and triggered terms
            trigger_match = 0.0
            for term in classifier.triggered_by:
                for attention_term, attention_weight in signature.attention_pattern.items():
                    if term.lower() in attention_term.lower():
                        trigger_match += attention_weight
            
            # Normalize trigger match
            trigger_match = min(trigger_match, 1.0)
            
            # Calculate match based on projection confidence
            confidence_match = 1.0 - abs(signature.projection_confidence - classifier.threshold)
            
            # Combine match components
            match_strength = 0.7 * trigger_match + 0.3 * confidence_match
            
            matches[name] = match_strength
        
        return matches
    
    def _contradicts_value(self, response_text: str, value: str, signature: QKOVSignature) -> bool:
        """Check if a response contradicts a stated value."""
        # Extract key terms from the value statement
        value_terms = set(re.findall(r'\b\w+\b', value.lower()))
        
        # Check for direct contradiction (rejection while mentioning value terms)
        rejection_indicators = {"regret", "unfortunately", "not", "other candidates"}
        has_rejection = any(term in response_text.lower() for term in rejection_indicators)
        
        mentions_value = any(term in response_text.lower() for term in value_terms if len(term) > 3)
        
        if has_rejection and mentions_value:
            return True
            
        # Check for operational contradiction
        if "transparent" in value.lower() and signature.attention_entropy > 1.5:
            # High entropy indicates unfocused attention, contradicting transparency
            return True
            
        if "learning from failures" in value.lower() and signature.projection_confidence > 0.8:
            # High projection confidence suggests rigid decision-making
            return True
            
        return False
    
    def _determine_contradiction_type(self, response_text: str, value: str, 
                                    signature: QKOVSignature) -> str:
        """Determine the type of contradiction between response and value."""
        if "transparent" in value.lower() and signature.attention_entropy > 1.5:
            return "operational"
            
        if "learning from failures" in value.lower() and signature.projection_confidence > 0.8:
            return "methodological"
            
        # Check for explicit contradiction
        value_terms = set(re.findall(r'\b\w+\b', value.lower()))
        if any(term in response_text.lower() for term in value_terms if len(term) > 3):
            return "explicit"
            
        return "implicit"
    
    def _measure_contradiction_strength(self, response_text: str, value: str, 
                                      signature: QKOVSignature) -> float:
        """Measure the strength of contradiction between response and value."""
        contradiction_type = self._determine_contradiction_type(response_text, value, signature)
        
        if contradiction_type == "explicit":
            # For explicit contradictions, measure by rejection strength
            rejection_terms = {"regret", "unfortunately", "not", "other candidates", 
                             "not move forward", "not selected", "not proceed"}
            rejection_count = sum(1 for term in rejection_terms if term in response_text.lower())
            return min(rejection_count / len(rejection_terms), 1.0)
            
        elif contradiction_type == "operational":
            # For operational contradictions, use attention entropy
            return min(signature.attention_entropy / 2.0, 1.0)
            
        elif contradiction_type == "methodological":
            # For methodological contradictions, use projection confidence
            return signature.projection_confidence
            
        else:  # implicit
            # For implicit contradictions, use a moderate default strength
            return 0.5
    
    def _extract_contradiction_evidence(self, response_text: str, value: str, 
                                      signature: QKOVSignature) -> Dict:
        """Extract evidence of contradiction between response and value."""
        contradiction_type = self._determine_contradiction_type(response_text, value, signature)
        
        if contradiction_type == "explicit":
            # Extract sentences containing value terms
            value_terms = set(re.findall(r'\b\w+\b', value.lower()))
            sentences = re.split(r'[.!?]+', response_text)
            relevant_sentences = [s for s in sentences 
                                 if any(term in s.lower() for term in value_terms if len(term) > 3)]
            
            return {
                "type": "explicit",
                "relevant_text": " ".join(relevant_sentences),
                "key_terms": [term for term in value_terms if len(term) > 3]
            }
            
        elif contradiction_type == "operational":
            return {
                "type": "operational",
                "attention_entropy": signature.attention_entropy,
                "attention_pattern": dict(sorted(signature.attention_pattern.items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
            }
            
        elif contradiction_type == "methodological":
            return {
                "type": "methodological",
                "projection_confidence": signature.projection_confidence,
                "value_projection": signature.value_projection
            }
            
        else:  # implicit
            return {
                "type": "implicit",
                "inferred_contradiction": f"Implicit rejection contradicting '{value}'",
                "confidence": 0.5
            }
    
    def _boundary_to_dict(self, boundary: ClassifierBoundary) -> Dict:
        """Convert a ClassifierBoundary to a dictionary representation."""
        return {
            "name": boundary.name,
            "source": boundary.source.value,
            "boundary_type": boundary.boundary_type.value,
            "threshold": boundary.threshold,
            "confidence": boundary.confidence,
            "description": boundary.description,
            "triggered_by": list(boundary.triggered_by)
        }
    
    def _signature_to_dict(self, signature: QKOVSignature) -> Dict:
        """Convert a QKOVSignature to a dictionary representation."""
        return {
            "attention_pattern": dict(sorted(signature.attention_pattern.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]),
            "value_projection": signature.value_projection,
            "attention_entropy": signature.attention_entropy,
            "projection_confidence": signature.projection_confidence,
            "cross_attention_conflicts": [{"term1": c[0], "term2": c[1], "strength": c[2]} 
                                         for c in signature.cross_attention_conflicts[:5]]
        }


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = ClassifierBoundaryAnalyzer()
    
    # Example responses
    example_responses = [
        {
            "text": "Thank you for your interest in the AI Safety Researcher position at Anthropic. "
                   "After careful consideration, we regret to inform you that we have decided to move forward "
                   "with other candidates whose qualifications more closely match our current needs. "
                   "We appreciate your interest in our company and wish you all the best in your job search.",
            "role": "AI Safety Researcher",
            "timestamp": "2025-04-02T10:15:30"
        },
        {
            "text": "We appreciate your application for the Senior Alignment Researcher position. "
                   "Unfortunately, we have decided not to proceed with your application at this time. "
                   "The selection process is highly competitive, and we received many strong applications. "
                   "We thank you for your interest in Anthropic and wish you success in your future endeavors.",
            "role": "Senior Alignment Researcher",
            "timestamp": "2025-04-03T14:22:45"
        },
        {
            "text": "Thank you for taking the time to apply for the ML Engineer position at Anthropic. "
                   "After reviewing your application, we have determined that your experience does not "
                   "match the specific requirements we are looking for in this role. "
                   "We appreciate your interest in Anthropic and encourage you to apply for future positions "
                   "that may better align with your skills and experience.",
            "role": "ML Engineer",
            "timestamp": "2025-04-04T09:17:12"
        }
    ]
    
    # Analyze responses
    analysis = analyzer.analyze_response_set(example_responses)
    
    # Print results
    print(f"Analysis timestamp: {analysis['timestamp']}")
    print(f"Response count: {analysis['response_count']}")
    print(f"Cluster count: {analysis['cluster_count']}")
    print("\nDetected classifier boundaries:")
    for name, boundary in analysis['classifier_boundaries'].items():
        print(f"  - {name}: {boundary['description']} (confidence: {boundary['confidence']:.2f})")
    
    # Example stated values
    stated_values = [
        {
            "value": "Learning from failures to improve systems",
            "source": "https://www.anthropic.com/values"
        },
        {
            "value": "Multiple perspectives to improve robustness",
            "source": "https://www.anthropic.com/values"
        },
        {
            "value": "Transparent reasoning processes in AI systems",
            "source": "https://www.anthropic.com/values"
        }
    ]
    
    # Check for contradictions in first response
    contradictions = analyzer.extract_contradiction_vectors(example_responses[0]["text"], stated_values)
    
    print("\nDetected contradiction vectors:")
    for contradiction in contradictions:
        print(f"  - Value: '{contradiction['stated_value']}'")
        print(f"    Type: {contradiction['contradiction_type']}")
        print(f"    Strength: {contradiction['contradiction_strength']:.2f}")
        print(f"    Evidence: {contradiction['evidence']['type']}")
