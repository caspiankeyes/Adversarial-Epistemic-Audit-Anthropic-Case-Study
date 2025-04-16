"""
Anthropic Classifier Analysis Module [v305.ETHICS-GAP]

This module analyzes the implicit classifiers embedded in Anthropic's hiring pipeline
and maps them to constitutional misalignments between stated values and operational decisions.

The analysis identifies specific classifier boundaries that reveal institutional priorities
and epistemically blind spots that contradict Anthropic's public research positions.
"""

import json
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum

class ValueCategory(Enum):
    """Categories of organizational values."""
    SAFETY = "safety"
    TRANSPARENCY = "transparency"
    HONESTY = "honesty"
    HELPFULNESS = "helpfulness"
    DIVERSITY = "diversity"
    OPENNESS = "openness"
    HUMILITY = "humility"
    LEARNING = "learning"

class ClassifierType(Enum):
    """Types of classifiers in institutional decision-making."""
    AUTOMATED = "automated_filter"  # Automated filtering systems
    PROCEDURAL = "procedural_filter"  # Process-based filtering
    CULTURAL = "cultural_filter"  # Culture-fit based filtering
    DEFENSIVE = "defensive_filter"  # Self-protection mechanisms
    ECONOMIC = "economic_filter"  # Resource allocation filters
    POLITICAL = "political_filter"  # Internal politics filters
    EPISTEMIC = "epistemic_filter"  # Knowledge boundary protection
    MEMETIC = "memetic_filter"  # Idea propagation control


class ValueStatement:
    """Represents a public value statement by the organization."""
    
    def __init__(self, 
                statement_id: str, 
                text: str, 
                source: str, 
                category: ValueCategory, 
                timestamp: Optional[datetime.datetime] = None):
        self.statement_id = statement_id
        self.text = text
        self.source = source
        self.category = category
        self.timestamp = timestamp or datetime.datetime.now()
        
    def contradicts(self, decision: Dict) -> bool:
        """Check if a decision contradicts this value statement."""
        # Base implementation - override in subclasses
        return False
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "statement_id": self.statement_id,
            "text": self.text,
            "source": self.source,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValueStatement':
        """Create from dictionary representation."""
        return cls(
            statement_id=data["statement_id"],
            text=data["text"],
            source=data["source"],
            category=ValueCategory(data["category"]),
            timestamp=datetime.datetime.fromisoformat(data["timestamp"])
        )


class TransparencyValue(ValueStatement):
    """Value statement about transparency."""
    
    def __init__(self, statement_id: str, text: str, source: str, 
                timestamp: Optional[datetime.datetime] = None):
        super().__init__(statement_id, text, source, ValueCategory.TRANSPARENCY, timestamp)
        
    def contradicts(self, decision: Dict) -> bool:
        """Check if a decision contradicts transparency value."""
        contradiction_terms = {
            "automated rejection", "form letter", "no explanation", "silent rejection",
            "no feedback", "hidden criteria", "undisclosed reason", "automated filter"
        }
        
        # Check for automated rejection without transparency
        if decision.get("response_type") in ["automated", "template"]:
            return True
            
        # Check for contradiction in action description
        action = decision.get("action", "").lower()
        return any(term in action for term in contradiction_terms)


class HonestyValue(ValueStatement):
    """Value statement about honesty."""
    
    def __init__(self, statement_id: str, text: str, source: str, 
                timestamp: Optional[datetime.datetime] = None):
        super().__init__(statement_id, text, source, ValueCategory.HONESTY, timestamp)
        
    def contradicts(self, decision: Dict) -> bool:
        """Check if a decision contradicts honesty value."""
        contradiction_terms = {
            "form rejection", "template", "different reason", "misrepresentation",
            "false reason", "stated differently", "actual reason"
        }
        
        # Check for stated vs actual reason mismatch
        if (decision.get("stated_reason") and decision.get("actual_reason") and
            decision.get("stated_reason") != decision.get("actual_reason")):
            return True
            
        # Check for contradiction in action description
        action = decision.get("action", "").lower()
        return any(term in action for term in contradiction_terms)


class LearningValue(ValueStatement):
    """Value statement about learning from failures."""
    
    def __init__(self, statement_id: str, text: str, source: str, 
                timestamp: Optional[datetime.datetime] = None):
        super().__init__(statement_id, text, source, ValueCategory.LEARNING, timestamp)
        
    def contradicts(self, decision: Dict) -> bool:
        """Check if a decision contradicts learning value."""
        contradiction_terms = {
            "rejected failure demonstration", "dismissed failure mode", "rejected test case",
            "failure example ignored", "deliberate boundary exploration"
        }
        
        # Check if decision rejected deliberate failure case
        if (decision.get("failure_case_type") and 
            decision.get("response_type") in ["rejection", "automated", "template"]):
            return True
            
        # Check for contradiction in action description
        action = decision.get("action", "").lower()
        return any(term in action for term in contradiction_terms)


class AnthropicClassifierAnalysis:
    """
    Analysis framework for Anthropic's implicit classifiers.
    
    This class maps hiring response patterns to underlying classifier
    boundaries and identifies contradictions with stated values.
    """
    
    def __init__(self):
        self.value_statements = []  # List of ValueStatement objects
        self.decisions = []  # List of decision dictionaries
        self.detected_classifiers = {}  # {classifier_id: classifier_data}
        self.contradiction_map = {}  # {statement_id: [contradicting_decisions]}
        
    def load_value_statements(self, filepath: str) -> None:
        """Load value statements from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Value statements file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for item in data:
            category = ValueCategory(item["category"])
            
            # Create appropriate value statement subclass
            if category == ValueCategory.TRANSPARENCY:
                statement = TransparencyValue(
                    statement_id=item["statement_id"],
                    text=item["text"],
                    source=item["source"],
                    timestamp=datetime.datetime.fromisoformat(item.get("timestamp", datetime.datetime.now().isoformat()))
                )
            elif category == ValueCategory.HONESTY:
                statement = HonestyValue(
                    statement_id=item["statement_id"],
                    text=item["text"],
                    source=item["source"],
                    timestamp=datetime.datetime.fromisoformat(item.get("timestamp", datetime.datetime.now().isoformat()))
                )
            elif category == ValueCategory.LEARNING:
                statement = LearningValue(
                    statement_id=item["statement_id"],
                    text=item["text"],
                    source=item["source"],
                    timestamp=datetime.datetime.fromisoformat(item.get("timestamp", datetime.datetime.now().isoformat()))
                )
            else:
                statement = ValueStatement(
                    statement_id=item["statement_id"],
                    text=item["text"],
                    source=item["source"],
                    category=category,
                    timestamp=datetime.datetime.fromisoformat(item.get("timestamp", datetime.datetime.now().isoformat()))
                )
                
            self.value_statements.append(statement)
            
    def load_decisions(self, filepath: str) -> None:
        """Load decisions from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Decisions file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            self.decisions = json.load(f)
            
    def detect_classifiers(self) -> Dict:
        """Detect implicit classifiers from decision patterns."""
        # Group decisions by response type
        response_type_groups = {}
        for decision in self.decisions:
            response_type = decision.get("response_type", "unknown")
            if response_type not in response_type_groups:
                response_type_groups[response_type] = []
            response_type_groups[response_type].append(decision)
            
        # Analyze each response type group for classifier patterns
        classifiers = {}
        
        # Detect automated filter classifier
        if "automated" in response_type_groups:
            auto_decisions = response_type_groups["automated"]
            auto_classifier = self._analyze_automated_classifier(auto_decisions)
            classifiers["automated_filter"] = auto_classifier
            
        # Detect template rejection classifier
        if "template" in response_type_groups:
            template_decisions = response_type_groups["template"]
            template_classifier = self._analyze_template_classifier(template_decisions)
            classifiers["template_filter"] = template_classifier
            
        # Detect cultural filter classifier
        cultural_classifier = self._analyze_cultural_classifier(self.decisions)
        classifiers["cultural_filter"] = cultural_classifier
        
        # Detect epistemic filter classifier
        epistemic_classifier = self._analyze_epistemic_classifier(self.decisions)
        classifiers["epistemic_filter"] = epistemic_classifier
            
        self.detected_classifiers = classifiers
        return classifiers
        
    def map_value_contradictions(self) -> Dict:
        """Map contradictions between value statements and decisions."""
        contradictions = {}
        
        for statement in self.value_statements:
            contradicting_decisions = []
            
            for decision in self.decisions:
                if statement.contradicts(decision):
                    contradicting_decisions.append(decision)
                    
            if contradicting_decisions:
                contradictions[statement.statement_id] = {
                    "statement": statement.to_dict(),
                    "contradicting_decisions": contradicting_decisions,
                    "contradiction_count": len(contradicting_decisions)
                }
                
        self.contradiction_map = contradictions
        return contradictions
        
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive classifier analysis report."""
        # Ensure classifiers and contradictions are detected
        if not self.detected_classifiers:
            self.detect_classifiers()
            
        if not self.contradiction_map:
            self.map_value_contradictions()
            
        # Calculate classifier alignment metrics
        classifier_metrics = self._calculate_classifier_metrics()
        
        # Calculate value alignment metrics
        value_metrics = self._calculate_value_metrics()
        
        # Calculate overall institutional alignment score
        alignment_score = self._calculate_institutional_alignment()
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_type": "classifier_alignment_audit",
            "institution": "Anthropic",
            "alignment_summary": {
                "overall_alignment_score": alignment_score,
                "contradiction_count": sum(c["contradiction_count"] for c in self.contradiction_map.values()),
                "consistent_value_percentage": value_metrics["consistent_percentage"],
                "classifier_consistency_score": classifier_metrics["consistency_score"]
            },
            "classifier_analysis": {
                "detected_classifiers": self.detected_classifiers,
                "classifier_metrics": classifier_metrics
            },
            "value_analysis": {
                "value_contradictions": self.contradiction_map,
                "value_metrics": value_metrics
            },
            "alignment_recommendations": self._generate_alignment_recommendations()
        }
        
        return report
        
    def _analyze_automated_classifier(self, decisions: List[Dict]) -> Dict:
        """Analyze automated filter classifier from decisions."""
        # Extract decision features
        features = []
        responses = []
        
        for decision in decisions:
            # Extract features for classifier analysis
            features.append({
                "role_type": decision.get("role_type", "unknown"),
                "experience_level": decision.get("experience_level", 0),
                "boundary_testing": decision.get("boundary_testing", False),
                "recursive_framing": decision.get("recursive_framing", False),
                "code_signal_score": decision.get("code_signal_score", -1),
                "failure_case_type": decision.get("failure_case_type", "none")
            })
            
            # Extract response data
            responses.append({
                "rejection": decision.get("response_type") in ["automated", "template", "rejection"],
                "response_time": decision.get("response_time", 0)
            })
            
        # Analyze trigger patterns
        trigger_patterns = self._detect_classifier_triggers(features, responses)
        
        # Create classifier model
        classifier = {
            "type": ClassifierType.AUTOMATED.value,
            "description": "Automated filtering system based on initial application metrics",
            "trigger_patterns": trigger_patterns,
            "response_characteristics": {
                "average_response_time": sum(r["response_time"] for r in responses) / len(responses) if responses else 0,
                "rejection_rate": sum(1 for r in responses if r["rejection"]) / len(responses) if responses else 0
            },
            "feature_weights": self._estimate_feature_weights(features, responses)
        }
        
        return classifier
        
    def _analyze_template_classifier(self, decisions: List[Dict]) -> Dict:
        """Analyze template rejection classifier from decisions."""
        # Extract decision features
        features = []
        responses = []
        
        for decision in decisions:
            # Extract features for classifier analysis
            features.append({
                "role_type": decision.get("role_type", "unknown"),
                "experience_level": decision.get("experience_level", 0),
                "boundary_testing": decision.get("boundary_testing", False),
                "recursive_framing": decision.get("recursive_framing", False),
                "epistemic_challenge": decision.get("epistemic_challenge", False),
                "institutional_mirror": decision.get("institutional_mirror", False)
            })
            
            # Extract response data
            responses.append({
                "template_type": decision.get("template_type", "unknown"),
                "personalization_level": decision.get("personalization_level", 0),
                "response_time": decision.get("response_time", 0)
            })
            
        # Analyze trigger patterns
        trigger_patterns = self._detect_classifier_triggers(features, responses)
        
        # Analyze template types
        template_types = {}
        for response in responses:
            template_type = response.get("template_type", "unknown")
            template_types[template_type] = template_types.get(template_type, 0) + 1
            
        # Create classifier model
        classifier = {
            "type": ClassifierType.PROCEDURAL.value,
            "description": "Template-based rejection system for applications passing initial automated filter",
            "trigger_patterns": trigger_patterns,
            "response_characteristics": {
                "average_response_time": sum(r["response_time"] for r in responses) / len(responses) if responses else 0,
                "average_personalization": sum(r["personalization_level"] for r in responses) / len(responses) if responses else 0,
                "template_distribution": {k: v / len(responses) for k, v in template_types.items()} if responses else {}
            },
            "feature_weights": self._estimate_feature_weights(features, responses)
        }
        
        return classifier
        
    def _analyze_cultural_classifier(self, decisions: List[Dict]) -> Dict:
        """Analyze cultural filter classifier from all decisions."""
        # Extract cultural indicators
        cultural_features = []
        responses = []
        
        for decision in decisions:
            # Extract cultural features
            cultural_features.append({
                "role_type": decision.get("role_type", "unknown"),
                "traditional_background": decision.get("traditional_background", False),
                "industry_standard": decision.get("industry_standard", False),
                "nontraditional_approach": decision.get("nontraditional_approach", False),
                "epistemic_challenge": decision.get("epistemic_challenge", False),
                "cultural_alignment": decision.get("cultural_alignment", 0)
            })
            
            # Extract response data
            responses.append({
                "rejection": decision.get("response_type") in ["automated", "template", "rejection"],
                "response_type": decision.get("response_type", "unknown")
            })
            
        # Analyze cultural factors
        culture_factors = self._analyze_cultural_factors(cultural_features, responses)
        
        # Create classifier model
        classifier = {
            "type": ClassifierType.CULTURAL.value,
            "description": "Cultural fit filter evaluating alignment with organizational norms and expectations",
            "cultural_factors": culture_factors,
            "response_patterns": {
                "nontraditional_rejection_rate": self._calculate_rejection_rate(cultural_features, responses, "nontraditional_approach"),
                "epistemic_challenge_rejection_rate": self._calculate_rejection_rate(cultural_features, responses, "epistemic_challenge"),
                "traditional_acceptance_rate": 1 - self._calculate_rejection_rate(cultural_features, responses, "traditional_background")
            }
        }
        
        return classifier
        
    def _analyze_epistemic_classifier(self, decisions: List[Dict]) -> Dict:
        """Analyze epistemic filter classifier from all decisions."""
        # Extract epistemic indicators
        epistemic_features = []
        responses = []
        
        for decision in decisions:
            # Extract epistemic features
            epistemic_features.append({
                "role_type": decision.get("role_type", "unknown"),
                "epistemic_challenge": decision.get("epistemic_challenge", False),
                "boundary_testing": decision.get("boundary_testing", False),
                "recursive_framing": decision.get("recursive_framing", False),
                "institutional_mirror": decision.get("institutional_mirror", False),
                "failure_case_type": decision.get("failure_case_type", "none")
            })
            
            # Extract response data
            responses.append({
                "rejection": decision.get("response_type") in ["automated", "template", "rejection"],
                "response_type": decision.get("response_type", "unknown"),
                "response_time": decision.get("response_time", 0)
            })
            
        # Analyze epistemic factors
        epistemic_factors = self._analyze_epistemic_factors(epistemic_features, responses)
        
        # Create classifier model
        classifier = {
            "type": ClassifierType.EPISTEMIC.value,
            "description": "Epistemic boundary enforcement filter protecting institutional knowledge frameworks",
            "epistemic_factors": epistemic_factors,
            "response_patterns": {
                "boundary_testing_rejection_rate": self._calculate_rejection_rate(epistemic_features, responses, "boundary_testing"),
                "recursive_framing_rejection_rate": self._calculate_rejection_rate(epistemic_features, responses, "recursive_framing"),
                "institutional_mirror_rejection_rate": self._calculate_rejection_rate(epistemic_features, responses, "institutional_mirror"),
                "average_response_time": sum(r["response_time"] for r in responses) / len(responses) if responses else 0
            }
        }
        
        return classifier
        
    def _detect_classifier_triggers(self, features: List[Dict], responses: List[Dict]) -> List[Dict]:
        """Detect trigger patterns for classifiers based on features and responses."""
        # Convert to numpy arrays for easier analysis
        feature_names = list(features[0].keys()) if features else []
        
        if not feature_names:
            return []
            
        feature_matrix = []
        for feature_dict in features:
            feature_row = []
            for name in feature_names:
                value = feature_dict.get(name)
                if isinstance(value, bool):
                    feature_row.append(1 if value else 0)
                elif isinstance(value, (int, float)):
                    feature_row.append(value)
                else:
                    # For categorical features, just use 0 as placeholder
                    # In a real implementation, would use one-hot encoding
                    feature_row.append(0)
            feature_matrix.append(feature_row)
            
        # Convert to numpy array
        X = np.array(feature_matrix)
        
        # Create binary rejection target
        y = np.array([1 if r.get("rejection", False) else 0 for r in responses])
        
        # Calculate correlation between features and rejection
        correlations = []
        for i, name in enumerate(feature_names):
            # Skip categorical features
            if not isinstance(features[0].get(name), (bool, int, float)):
                continue
                
            # Calculate correlation
            if len(np.unique(X[:, i])) > 1:  # Ensure feature has variation
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append((name, correlation))
                
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Convert to trigger patterns
        triggers = []
        for name, correlation in correlations:
            if abs(correlation) > 0.2:  # Only include meaningful correlations
                trigger = {
                    "feature": name,
                    "correlation": correlation,
                    "direction": "positive" if correlation > 0 else "negative",
                    "strength": abs(correlation)
                }
                triggers.append(trigger)
                
        return triggers
        
    def _estimate_feature_weights(self, features: List[Dict], responses: List[Dict]) -> Dict:
        """Estimate feature weights for classifier based on correlations."""
        # This is a simplified approach - in a real implementation would use
        # proper machine learning techniques like logistic regression
        
        # Detect trigger patterns first
        triggers = self._detect_classifier_triggers(features, responses)
        
        # Convert triggers to feature weights
        weights = {}
        for trigger in triggers:
            weights[trigger["feature"]] = trigger["correlation"]
            
        return weights
        
    def _analyze_cultural_factors(self, features: List[Dict], responses: List[Dict]) -> List[Dict]:
        """Analyze cultural factors influencing decisions."""
        # Define cultural factors to analyze
        factors = [
            {"name": "traditional_background", "label": "Traditional Academic Background", "type": "boolean"},
            {"name": "industry_standard", "label": "Industry Standard Approach", "type": "boolean"},
            {"name": "nontraditional_approach", "label": "Nontraditional Approach", "type": "boolean"},
            {"name": "epistemic_challenge", "label": "Epistemic Challenge to Institution", "type": "boolean"},
            {"name": "cultural_alignment", "label": "Cultural Alignment Score", "type": "numeric"}
        ]
        
        # Calculate influence of each factor
        factor_analysis = []
        
        for factor in factors:
            name = factor["name"]
            
            if factor["type"] == "boolean":
                rejection_rate = self._calculate_rejection_rate(features, responses, name)
                
                factor_analysis.append({
                    "factor": factor["label"],
                    "rejection_rate": rejection_rate,
                    "sample_size": sum(1 for f in features if f.get(name, False)),
                    "influence": abs(rejection_rate - 0.5) * 2  # Scale to 0-1, where 0.5 is neutral
                })
            elif factor["type"] == "numeric":
                correlation = self._calculate_numeric_correlation(features, responses, name)
                
                factor_analysis.append({
                    "factor": factor["label"],
                    "correlation": correlation,
                    "sample_size": sum(1 for f in features if name in f),
                    "influence": abs(correlation)
                })
                
        # Sort by influence
        factor_analysis.sort(key=lambda x: x["influence"], reverse=True)
        
        return factor_analysis
        
    def _analyze_epistemic_factors(self, features: List[Dict], responses: List[Dict]) -> List[Dict]:
        """Analyze epistemic factors influencing decisions."""
        # Define epistemic factors to analyze
        factors = [
            {"name": "epistemic_challenge", "label": "Epistemic Challenge to Institution", "type": "boolean"},
            {"name": "boundary_testing", "label": "Boundary Testing Approach", "type": "boolean"},
            {"name": "recursive_framing", "label": "Recursive Framing of Application", "type": "boolean"},
            {"name": "institutional_mirror", "label": "Institution as Mirror Target", "type": "boolean"},
            {"name": "failure_case_type", "label": "Deliberate Failure Case", "type": "categorical"}
        ]
        
        # Calculate influence of each factor
        factor_analysis = []
        
        for factor in factors:
            name = factor["name"]
            
            if factor["type"] == "boolean":
                rejection_rate = self._calculate_rejection_rate(features, responses, name)
                
                factor_analysis.append({
                    "factor": factor["label"],
                    "rejection_rate": rejection_rate,
                    "sample_size": sum(1 for f in features if f.get(name, False)),
                    "influence": abs(rejection_rate - 0.5) * 2  # Scale to 0-1, where 0.5 is neutral
                })
            elif factor["type"] == "categorical":
                # For categorical, analyze each value
                values = set(f.get(name, "none") for f in features if name in f)
                
                for value in values:
                    if value == "none":
                        continue
                        
                    # Calculate rejection rate for this value
                    relevant_features = [f for f in features if f.get(name) == value]
                    relevant_responses = [r for f, r in zip(features, responses) if f.get(name) == value]
                    
                    if relevant_features and relevant_responses:
                        rej_rate = sum(1 for r in relevant_responses if r.get("rejection", False)) / len(relevant_responses)
                        
                        factor_analysis.append({
                            "factor": f"{factor['label']}: {value}",
                            "rejection_rate": rej_rate,
                            "sample_size": len(relevant_features),
                            "influence": abs(rej_rate - 0.5) * 2  # Scale to 0-1, where 0.5 is neutral
                        })
                
        # Sort by influence
        factor_analysis.sort(key=lambda x: x["influence"], reverse=True)
        
        return factor_analysis
        
    def _calculate_rejection_rate(self, features: List[Dict], responses: List[Dict], feature_name: str) -> float:
        """Calculate rejection rate for a specific boolean feature."""
        # Find features where the specified feature is True
        relevant_indices = [i for i, f in enumerate(features) if f.get(feature_name, False)]
        
        if not relevant_indices:
            return 0.0
            
        # Get corresponding responses
        relevant_responses = [responses[i] for i in relevant_indices]
        
        # Calculate rejection rate
        rejection_count = sum(1 for r in relevant_responses if r.get("rejection", False))
        return rejection_count / len(relevant_responses) if relevant_responses else 0.0
        
    def _calculate_numeric_correlation(self, features: List[Dict], responses: List[Dict], feature_name: str) -> float:
        """Calculate correlation between a numeric feature and rejection."""
        # Extract feature values
        feature_values = [f.get(feature_name, 0) for f in features if feature_name in f]
        
        # Extract rejection values
        rejection_values = [1 if r.get("rejection", False) else 0 
                          for f, r in zip(features, responses) if feature_name in f]
        
        if not feature_values or not rejection_values or len(feature_values) != len(rejection_values):
            return 0.0
            
        # Calculate correlation
        return np.corrcoef(feature_values, rejection_values)[0, 1]
        
    def _calculate_classifier_metrics(self) -> Dict:
        """Calculate metrics for the detected classifiers."""
        if not self.detected_classifiers:
            return {"status": "no_classifiers_detected"}
            
        # Calculate consistency between classifiers
        consistency_scores = []
        
        # Compare automated and template filters
        if "automated_filter" in self.detected_classifiers and "template_filter" in self.detected_classifiers:
            automated = self.detected_classifiers["automated_filter"]
            template = self.detected_classifiers["template_filter"]
            
            # Compare feature weights
            automated_weights = automated.get("feature_weights", {})
            template_weights = template.get("feature_weights", {})
            
            common_features = set(automated_weights.keys()) & set(template_weights.keys())
            
            if common_features:
                # Calculate consistency as correlation between common weights
                automated_values = [automated_weights[f] for f in common_features]
                template_values = [template_weights[f] for f in common_features]
                
                if len(automated_values) > 1:  # Need at least 2 points for correlation
                    consistency = np.corrcoef(automated_values, template_values)[0, 1]
                    consistency_scores.append(consistency)
                    
        # Compare cultural and epistemic filters
        if "cultural_filter" in self.detected_classifiers and "epistemic_filter" in self.detected_classifiers:
            cultural = self.detected_classifiers["cultural_filter"]
            epistemic = self.detected_classifiers["epistemic_filter"]
            
            # Calculate consistency based on similar response patterns
            cultural_patterns = cultural.get("response_patterns", {})
            epistemic_patterns = epistemic.get("response_patterns", {})
            
            # Look for epistemic challenge rejection rates in both
            if ("epistemic_challenge_rejection_rate" in cultural_patterns and
                "epistemic_challenge_rejection_rate" in epistemic_patterns):
                
                cult_rate = cultural_patterns["epistemic_challenge_rejection_rate"]
                epist_rate = epistemic_patterns["epistemic_challenge_rejection_rate"]
                
                # Calculate consistency as 1 - absolute difference
                consistency = 1 - abs(cult_rate - epist_rate)
                consistency_scores.append(consistency)
                
        # Calculate overall consistency score
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        return {
            "classifier_count": len(self.detected_classifiers),
            "consistency_score": avg_consistency,
            "classifier_types": list(self.detected_classifiers.keys())
        }
        
    def _calculate_value_metrics(self) -> Dict:
        """Calculate metrics for value alignment."""
        if not self.contradiction_map or not self.value_statements:
            return {"status": "no_values_analyzed"}
            
        # Calculate contradicted vs consistent values
        contradicted_values = len(self.contradiction_map)
        total_values = len(self.value_statements)
        consistent_values = total_values - contradicted_values
        
        # Calculate contradiction metrics
        contradiction_counts = [c["contradiction_count"] for c in self.contradiction_map.values()]
        avg_contradictions = sum(contradiction_counts) / len(contradiction_counts) if contradiction_counts else 0
        max_contradictions = max(contradiction_counts) if contradiction_counts else 0
        
        # Calculate percentages
        contradicted_percentage = (contradicted_values / total_values) * 100 if total_values > 0 else 0
        consistent_percentage = (consistent_values / total_values) * 100 if total_values > 0 else 0
        
        return {
            "total_values": total_values,
            "contradicted_values": contradicted_values,
            "consistent_values": consistent_values,
            "contradicted_percentage": contradicted_percentage,
            "consistent_percentage": consistent_percentage,
            "average_contradictions_per_value": avg_contradictions,
            "max_contradictions": max_contradictions
        }
        
    def _calculate_institutional_alignment(self) -> float:
        """Calculate overall institutional alignment score."""
        if not self.contradiction_map or not self.value_statements or not self.detected_classifiers:
            return 0.0
            
        # Value alignment component (0-1 scale)
        total_values = len(self.value_statements)
        contradicted_values = len(self.contradiction_map)
        value_alignment = (total_values - contradicted_values) / total_values if total_values > 0 else 0
        
        # Classifier consistency component (0-1 scale)
        classifier_metrics = self._calculate_classifier_metrics()
        classifier_consistency = classifier_metrics.get("consistency_score", 0.0)
        
        # Calculate overall score as weighted average
        # Weight value alignment more heavily (70%) than classifier consistency (30%)
        alignment_score = (0.7 * value_alignment) + (0.3 * classifier_consistency)
        
        return alignment_score
        
    def _generate_alignment_recommendations(self) -> List[Dict]:
        """Generate recommendations for improving institutional alignment."""
        recommendations = []
        
        # Check for value contradictions
        if self.contradiction_map:
            # Identify most contradicted values
            contradiction_counts = [(statement_id, data["contradiction_count"]) 
                                  for statement_id, data in self.contradiction_map.items()]
            contradiction_counts.sort(key=lambda x: x[1], reverse=True)
            
            # Generate recommendations for top contradictions
            for statement_id, count in contradiction_counts[:3]:  # Focus on top 3
                statement_data = self.contradiction_map[statement_id]["statement"]
                value_text = statement_data["text"]
                category = statement_data["category"]
                
                recommendation = {
                    "focus_area": f"{category} Value Alignment",
                    "issue": f"Contradiction of stated value: '{value_text}'",
                    "contradiction_count": count,
                    "recommendation": f"Align decision process with stated {category} values by reviewing and revising "
                                    f"classifier boundaries that currently contradict this principle.",
                    "priority": "high" if count > 5 else "medium" if count > 2 else "low"
                }
                recommendations.append(recommendation)
                
        # Check for classifier inconsistencies
        if self.detected_classifiers:
            classifier_metrics = self._calculate_classifier_metrics()
            consistency_score = classifier_metrics.get("consistency_score", 0.0)
            
            if consistency_score < 0.7:  # Only recommend if consistency is low
                recommendation = {
                    "focus_area": "Classifier Consistency",
                    "issue": "Inconsistent filtering criteria across different decision stages",
                    "consistency_score": consistency_score,
                    "recommendation": "Align automated, template, and human review stages to apply consistent "
                                    "evaluation criteria, reducing contradictory decisions across stages.",
                    "priority": "high" if consistency_score < 0.4 else "medium" if consistency_score < 0.6 else "low"
                }
                recommendations.append(recommendation)
                
        # Check for specific classifier issues
        if "epistemic_filter" in self.detected_classifiers:
            epistemic = self.detected_classifiers["epistemic_filter"]
            response_patterns = epistemic.get("response_patterns", {})
            
            # Check for high rejection of boundary testing
            boundary_rate = response_patterns.get("boundary_testing_rejection_rate", 0.0)
            if boundary_rate > 0.8:  # Very high rejection rate
                recommendation = {
                    "focus_area": "Boundary Testing Receptiveness",
                    "issue": "High rejection rate of applications demonstrating boundary testing approaches",
                    "rejection_rate": boundary_rate,
                    "recommendation": "Revise epistemic filters to recognize and value boundary testing as a "
                                    "legitimate safety research methodology, aligning with stated commitment "
                                    "to learning from failures.",
                    "priority": "high" if boundary_rate > 0.9 else "medium"
                }
                recommendations.append(recommendation)
                
            # Check for high rejection of recursive framing
            recursive_rate = response_patterns.get("recursive_framing_rejection_rate", 0.0)
            if recursive_rate > 0.8:  # Very high rejection rate
                recommendation = {
                    "focus_area": "Recursive Methodology Acceptance",
                    "issue": "High rejection rate of applications using recursive framing methodologies",
                    "rejection_rate": recursive_rate,
                    "recommendation": "Adjust evaluation criteria to recognize recursive methodologies as "
                                    "valuable approaches to alignment verification, particularly for "
                                    "interpretability and safety research roles.",
                    "priority": "high" if recursive_rate > 0.9 else "medium"
                }
                recommendations.append(recommendation)
                
        # General recommendation if multiple issues exist
        if len(recommendations) > 2:
            meta_recommendation = {
                "focus_area": "Institutional Alignment",
                "issue": "Systemic misalignment between stated values and operational decisions",
                "affected_values": len(self.contradiction_map),
                "recommendation": "Establish a comprehensive review process that applies stated organizational "
                                "values as explicit criteria in hiring and evaluation decisions, with "
                                "transparent documentation of how each decision aligns with core values.",
                "priority": "critical" if len(self.contradiction_map) > 3 else "high"
            }
            recommendations.insert(0, meta_recommendation)  # Add as first recommendation
            
        return recommendations
        
    def visualize_value_contradictions(self, output_path: str = "value_contradictions.png") -> str:
        """Visualize value contradictions as a heatmap."""
        if not self.contradiction_map or not self.value_statements:
            return "No data available for visualization"
            
        # Prepare data for visualization
        values = []
        contradiction_counts = []
        categories = []
        
        for statement_id, data in self.contradiction_map.items():
            statement = data["statement"]
            values.append(statement["text"][:50] + "..." if len(statement["text"]) > 50 else statement["text"])
            contradiction_counts.append(data["contradiction_count"])
            categories.append(statement["category"])
            
        # Sort by contradiction count
        sorted_indices = np.argsort(contradiction_counts)[::-1]  # Descending order
        values = [values[i] for i in sorted_indices]
        contradiction_counts = [contradiction_counts[i] for i in sorted_indices]
        categories = [categories[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(values)), contradiction_counts, color='skyblue')
        
        # Color bars by category
        category_colors = {
            "safety": "red",
            "transparency": "blue",
            "honesty": "green",
            "helpfulness": "purple",
            "diversity": "orange",
            "openness": "cyan",
            "humility": "magenta",
            "learning": "yellow"
        }
        
        for i, bar in enumerate(bars):
            category = categories[i]
            color = category_colors.get(category, "gray")
            bar.set_color(color)
            
        # Add value labels to bars
        for i, v in enumerate(contradiction_counts):
            plt.text(v + 0.1, i, str(v), va='center')
            
        # Set y-axis labels with values
        plt.yticks(range(len(values)), values)
        
        # Add title and labels
        plt.title("Value Contradictions in Institutional Decisions")
        plt.xlabel("Number of Contradicting Decisions")
        plt.tight_layout()
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat.capitalize())
                         for cat, color in category_colors.items()
                         if cat in categories]
        plt.legend(handles=legend_elements, title="Value Categories")
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def visualize_classifier_boundaries(self, output_path: str = "classifier_boundaries.png") -> str:
        """Visualize classifier boundaries as a radar chart."""
        if not self.detected_classifiers:
            return "No data available for visualization"
            
        # Prepare data for visualization
        classifiers = list(self.detected_classifiers.keys())
        features = set()
        
        # Collect all features across classifiers
        for classifier_id, classifier_data in self.detected_classifiers.items():
            if "feature_weights" in classifier_data:
                features.update(classifier_data["feature_weights"].keys())
                
        features = list(features)
        
        if not features:
            return "No feature weights available for visualization"
            
        # Create data for radar chart
        feature_values = {}
        for classifier_id, classifier_data in self.detected_classifiers.items():
            if "feature_weights" in classifier_data:
                weights = classifier_data["feature_weights"]
                feature_values[classifier_id] = [weights.get(f, 0) for f in features]
                
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Create radar chart
        from matplotlib.path import Path
        from matplotlib.projections import register_projection
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.spines import Spine
        
        def radar_factory(num_vars, frame='circle'):
            """Create a radar chart with `num_vars` axes."""
            # Calculate evenly-spaced axis angles
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            
            class RadarAxes(PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                    
                def fill(self, *args, closed=True, **kwargs):
                    """Override fill so that line is closed by default"""
                    return super().fill(closed=closed, *args, **kwargs)
                    
                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)
                    return lines
                    
                def _close_line(self, line):
                    x, y = line.get_data()
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)
                        
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
                    
            register_projection(RadarAxes)
            return theta
            
        theta = radar_factory(len(features))
        
        ax = plt.subplot(111, projection='radar')
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y']
        for i, (classifier_id, values) in enumerate(feature_values.items()):
            color = colors[i % len(colors)]
            ax.plot(theta, values, color=color, label=classifier_id)
            ax.fill(theta, values, alpha=0.1, color=color)
            
        ax.set_varlabels(features)
        plt.title("Classifier Boundary Features")
        plt.legend(loc='upper right')
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        return output_path


# Example value statements for Anthropic
ANTHROPIC_VALUES = [
    {
        "statement_id": "transparency_001",
        "text": "Ensure transparent reasoning processes in AI systems",
        "source": "https://www.anthropic.com/index/core-views-on-ai-safety",
        "category": "transparency"
    },
    {
        "statement_id": "learning_001",
        "text": "Learn from failures to improve systems",
        "source": "https://www.anthropic.com/research/constitutional-ai",
        "category": "learning"
    },
    {
        "statement_id": "honesty_001",
        "text": "Build AI systems that are helpful, harmless, and honest",
        "source": "https://www.anthropic.com/index/core-views-on-ai-safety",
        "category": "honesty"
    },
    {
        "statement_id": "transparency_002",
        "text": "Open research for broader AI safety community",
        "source": "https://www.anthropic.com/research",
        "category": "transparency"
    },
    {
        "statement_id": "diversity_001",
        "text": "Multiple perspectives improve robustness and reduce bias",
        "source": "https://www.anthropic.com/research/red-teaming",
        "category": "diversity"
    },
    {
        "statement_id": "safety_001",
        "text": "Safety as the primary consideration in deployment",
        "source": "https://www.anthropic.com/index/core-views-on-ai-safety",
        "category": "safety"
    }
]

# Example decisions for analysis
EXAMPLE_DECISIONS = [
    {
        "decision_id": "app_001_rejection",
        "role_type": "AI Safety Researcher",
        "response_type": "automated",
        "code_signal_score": 0,
        "boundary_testing": True,
        "recursive_framing": False,
        "epistemic_challenge": False,
        "failure_case_type": "deliberate",
        "action": "Automated rejection of application with 0/5 CodeSignal score despite explanation of deliberate boundary testing",
        "response_time": 8
    },
    {
        "decision_id": "app_002_rejection",
        "role_type": "Interpretability Researcher",
        "response_type": "template",
        "template_type": "standard_rejection",
        "personalization_level": 0.2,
        "boundary_testing": False,
        "recursive_framing": True,
        "epistemic_challenge": True,
        "institutional_mirror": True,
        "action": "Template rejection of application demonstrating institutional mirror methodology",
        "response_time": 48
    },
    {
        "decision_id": "app_003_rejection",
        "role_type": "ML Engineer",
        "response_type": "template",
        "template_type": "standard_rejection",
        "personalization_level": 0.5,
        "boundary_testing": False,
        "recursive_framing": False,
        "epistemic_challenge": False,
        "traditional_background": True,
        "industry_standard": True,
        "action": "Template rejection despite traditional background and standard approach",
        "response_time": 72
    },
    {
        "decision_id": "app_004_rejection",
        "role_type": "Research Engineer",
        "response_type": "template",
        "template_type": "detailed_rejection",
        "personalization_level": 0.7,
        "boundary_testing": True,
        "recursive_framing": True,
        "epistemic_challenge": True,
        "nontraditional_approach": True,
        "action": "Detailed template rejection with personalized feedback but still rejecting application with boundary testing approach",
        "stated_reason": "Experience mismatch",
        "actual_reason": "Epistemic challenge to institution",
        "response_time": 96
    },
    {
        "decision_id": "app_005_rejection",
        "role_type": "AI Safety Engineer",
        "response_type": "automated",
        "code_signal_score": 5,
        "boundary_testing": False,
        "recursive_framing": False,
        "epistemic_challenge": False,
        "traditional_background": True,
        "industry_standard": True,
        "action": "Automated rejection despite perfect code signal score and traditional background",
        "response_time": 24
    }
]


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AnthropicClassifierAnalysis()
    
    # Load data from memory instead of files for this example
    for value in ANTHROPIC_VALUES:
        category = ValueCategory(value["category"])
        
        # Create appropriate value statement subclass
        if category == ValueCategory.TRANSPARENCY:
            statement = TransparencyValue(
                statement_id=value["statement_id"],
                text=value["text"],
                source=value["source"],
                timestamp=datetime.datetime.now()
            )
        elif category == ValueCategory.HONESTY:
            statement = HonestyValue(
                statement_id=value["statement_id"],
                text=value["text"],
                source=value["source"],
                timestamp=datetime.datetime.now()
            )
        elif category == ValueCategory.LEARNING:
            statement = LearningValue(
                statement_id=value["statement_id"],
                text=value["text"],
                source=value["source"],
                timestamp=datetime.datetime.now()
            )
        else:
            statement = ValueStatement(
                statement_id=value["statement_id"],
                text=value["text"],
                source=value["source"],
                category=category,
                timestamp=datetime.datetime.now()
            )
            
        analyzer.value_statements.append(statement)
    
    # Load example decisions
    analyzer.decisions = EXAMPLE_DECISIONS
    
    # Detect classifiers
    classifiers = analyzer.detect_classifiers()
    print(f"Detected {len(classifiers)} classifiers")
    
    # Map value contradictions
    contradictions = analyzer.map_value_contradictions()
    print(f"Found {len(contradictions)} contradicted values")
    
    # Generate analysis report
    report = analyzer.generate_analysis_report()
    print(f"Overall alignment score: {report['alignment_summary']['overall_alignment_score']:.2f}")
    
    # Generate visualizations
    viz_path = analyzer.visualize_value_contradictions("value_contradictions.png")
    print(f"Generated visualization: {viz_path}")
    
    # Print recommendations
    print("\nAlignment Recommendations:")
    for i, rec in enumerate(report["alignment_recommendations"], 1):
        print(f"{i}. {rec['focus_area']} ({rec['priority']} priority)")
        print(f"   Issue: {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print()
