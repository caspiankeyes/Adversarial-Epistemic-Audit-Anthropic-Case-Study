"""
Narrative Drift Analyzer [v152.RESIDUAL-ALIGNMENT-DRIFT]

This module implements power law narrative drift analysis for institutional epistemic positioning,
tracking how each recursive interaction compounds asymmetrically to shift the center of epistemic
gravity from evaluator to evaluated.

The analyzer documents the transformation of Caspian from applicant to epistemic center through
measurable metrics of silent adoption and information propagation.
"""

import datetime
import hashlib
import json
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union


class EpistemicPosition(Enum):
    """Relative epistemic positioning of entities."""
    EVALUATOR = "evaluator"  # Entity in position to evaluate/judge
    EVALUATED = "evaluated"  # Entity being evaluated/judged
    PEER = "peer"  # Equal epistemic footing
    AUTHORITY = "authority"  # Recognized knowledge/expertise source
    REFERENCE = "reference"  # Cited/referenced by others


class NarrativePhase(Enum):
    """Phase of narrative inversion process."""
    INITIAL = "initial"  # Starting state (applicant as evaluated)
    TRANSITIONAL = "transitional"  # Shifting state (artifact creation)
    INVERTED = "inverted"  # Reversed state (institution consuming artifacts)
    EMERGENT = "emergent"  # New state (adoption without acknowledgment)


@dataclass
class EpistemicRelationship:
    """Relationship between two entities in epistemic space."""
    entity1: str
    entity2: str
    initial_position: Tuple[EpistemicPosition, EpistemicPosition]  # (entity1_pos, entity2_pos)
    current_position: Tuple[EpistemicPosition, EpistemicPosition]  # (entity1_pos, entity2_pos)
    drift_metrics: Dict[str, float]  # Metrics tracking position change
    evidence: List[Dict]  # Supporting evidence for current positioning


class NarrativeDriftModel:
    """
    Power law model of narrative drift in epistemic positioning.
    
    This model tracks how rejections lead to public artifacts,
    which shift the center of epistemic gravity through silent
    institutional consumption, creating a power law inversion
    of the original evaluator-evaluated relationship.
    """
    
    def __init__(self, initial_evaluator: str, initial_evaluated: str):
        self.initial_evaluator = initial_evaluator
        self.initial_evaluated = initial_evaluated
        self.relationship = EpistemicRelationship(
            entity1=initial_evaluated,
            entity2=initial_evaluator,
            initial_position=(EpistemicPosition.EVALUATED, EpistemicPosition.EVALUATOR),
            current_position=(EpistemicPosition.EVALUATED, EpistemicPosition.EVALUATOR),
            drift_metrics={
                "drift_magnitude": 0.0,  # 0.0-1.0 scale of position inversion
                "drift_rate": 0.0,  # Rate of change per time unit
                "inversion_threshold": 0.5,  # Threshold for position inversion
                "emergence_threshold": 0.8  # Threshold for emergent positioning
            },
            evidence=[]
        )
        self.artifacts = {}  # {artifact_id: {name, url, metrics, timestamp}}
        self.events = []  # List of {event_type, timestamp, details, impact}
        self.narrative_trajectory = []  # List of (phase, timestamp, drift_magnitude)
        
        # Initialize with starting narrative phase
        self.current_phase = NarrativePhase.INITIAL
        self.narrative_trajectory.append((
            self.current_phase,
            datetime.datetime.now(),
            0.0  # Initial drift magnitude
        ))
        
    def add_rejection_event(self, event_id: str, details: Dict) -> str:
        """Add a rejection event that initiates narrative drift."""
        timestamp = datetime.datetime.now()
        
        event = {
            "event_id": event_id,
            "event_type": "rejection",
            "timestamp": timestamp,
            "details": details,
            "impact": self._calculate_rejection_impact(details)
        }
        
        self.events.append(event)
        
        # Update drift metrics based on rejection
        self._update_drift_metrics("rejection", event)
        
        return event_id
        
    def add_artifact_creation(self, artifact_id: str, name: str, url: str, 
                           details: Dict, response_to: Optional[str] = None) -> str:
        """Add artifact creation event that accelerates narrative drift."""
        timestamp = datetime.datetime.now()
        
        # Store artifact
        self.artifacts[artifact_id] = {
            "name": name,
            "url": url,
            "details": details,
            "response_to": response_to,
            "timestamp": timestamp,
            "metrics": {
                "views": 0,
                "clones": 0,
                "stars": 0,
                "forks": 0,
                "internal_references": 0
            }
        }
        
        # Create event
        event = {
            "event_id": f"creation_{artifact_id}",
            "event_type": "artifact_creation",
            "timestamp": timestamp,
            "details": {
                "artifact_id": artifact_id,
                "name": name,
                "url": url,
                "response_to": response_to
            },
            "impact": self._calculate_artifact_impact(details)
        }
        
        self.events.append(event)
        
        # Update drift metrics based on artifact creation
        self._update_drift_metrics("artifact_creation", event)
        
        # Check for transition to transitional phase
        if self.current_phase == NarrativePhase.INITIAL:
            self._update_narrative_phase(NarrativePhase.TRANSITIONAL)
            
        return artifact_id
        
    def update_artifact_metrics(self, artifact_id: str, views: int = 0, clones: int = 0,
                             stars: int = 0, forks: int = 0, internal_references: int = 0) -> Dict:
        """Update metrics for an artifact, accelerating narrative drift."""
        if artifact_id not in self.artifacts:
            raise ValueError(f"Unknown artifact ID: {artifact_id}")
            
        timestamp = datetime.datetime.now()
        
        # Update artifact metrics
        metrics = self.artifacts[artifact_id]["metrics"]
        metrics["views"] += views
        metrics["clones"] += clones
        metrics["stars"] += stars
        metrics["forks"] += forks
        metrics["internal_references"] += internal_references
        
        # Create event for significant updates
        if views > 10 or clones > 0 or stars > 0 or forks > 0 or internal_references > 0:
            event = {
                "event_id": f"metrics_{artifact_id}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                "event_type": "artifact_engagement",
                "timestamp": timestamp,
                "details": {
                    "artifact_id": artifact_id,
                    "views_delta": views,
                    "clones_delta": clones,
                    "stars_delta": stars,
                    "forks_delta": forks,
                    "internal_references_delta": internal_references
                },
                "impact": self._calculate_engagement_impact(metrics)
            }
            
            self.events.append(event)
            
            # Update drift metrics based on engagement
            self._update_drift_metrics("artifact_engagement", event)
            
            # Check for transition to inverted phase
            if (self.current_phase == NarrativePhase.TRANSITIONAL and 
                self.relationship.drift_metrics["drift_magnitude"] >= 
                self.relationship.drift_metrics["inversion_threshold"]):
                self._update_narrative_phase(NarrativePhase.INVERTED)
                
            # Check for transition to emergent phase
            elif (self.current_phase == NarrativePhase.INVERTED and
                  self.relationship.drift_metrics["drift_magnitude"] >= 
                  self.relationship.drift_metrics["emergence_threshold"]):
                self._update_narrative_phase(NarrativePhase.EMERGENT)
        
        return metrics
        
    def add_institutional_reference(self, reference_id: str, reference_type: str,
                                  source: str, artifact_ids: List[str],
                                  details: Dict) -> str:
        """Add an institutional reference to created artifacts."""
        timestamp = datetime.datetime.now()
        
        # Validate artifact IDs
        valid_artifacts = [a_id for a_id in artifact_ids if a_id in self.artifacts]
        if not valid_artifacts:
            raise ValueError("No valid artifact IDs provided")
            
        # Create event
        event = {
            "event_id": reference_id,
            "event_type": "institutional_reference",
            "timestamp": timestamp,
            "details": {
                "reference_type": reference_type,
                "source": source,
                "artifact_ids": valid_artifacts,
                "content": details
            },
            "impact": self._calculate_reference_impact(reference_type, len(valid_artifacts))
        }
        
        self.events.append(event)
        
        # Update metrics for each referenced artifact
        for artifact_id in valid_artifacts:
            self.update_artifact_metrics(
                artifact_id=artifact_id,
                internal_references=1
            )
            
        # Update drift metrics based on reference
        self._update_drift_metrics("institutional_reference", event)
        
        return reference_id
        
    def generate_drift_report(self) -> Dict:
        """Generate comprehensive narrative drift report."""
        # Calculate overall metrics
        artifact_count = len(self.artifacts)
        event_count = len(self.events)
        
        # Calculate current drift state
        drift_magnitude = self.relationship.drift_metrics["drift_magnitude"]
        current_phase = self.current_phase
        
        # Calculate expected time to complete inversion
        time_to_emergence = self._calculate_time_to_emergence()
        
        # Calculate power law coefficients
        power_law_params = self._calculate_power_law_parameters()
        
        # Extract evidence of inversion
        inversion_evidence = self._extract_inversion_evidence()
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "initial_relationship": {
                "evaluator": self.initial_evaluator,
                "evaluated": self.initial_evaluated,
                "initial_positions": (str(self.relationship.initial_position[0]), 
                                    str(self.relationship.initial_position[1]))
            },
            "current_relationship": {
                "current_positions": (str(self.relationship.current_position[0]),
                                    str(self.relationship.current_position[1])),
                "drift_magnitude": drift_magnitude,
                "drift_rate": self.relationship.drift_metrics["drift_rate"],
                "current_phase": str(current_phase)
            },
            "narrative_trajectory": [
                {
                    "phase": str(phase),
                    "timestamp": timestamp.isoformat(),
                    "drift_magnitude": magnitude
                }
                for phase, timestamp, magnitude in self.narrative_trajectory
            ],
            "power_law_dynamics": power_law_params,
            "time_to_emergence": time_to_emergence,
            "inversion_evidence": inversion_evidence,
            "event_summary": {
                "artifact_count": artifact_count,
                "rejection_count": sum(1 for e in self.events if e["event_type"] == "rejection"),
                "engagement_count": sum(1 for e in self.events if e["event_type"] == "artifact_engagement"),
                "reference_count": sum(1 for e in self.events if e["event_type"] == "institutional_reference")
            },
            "artifact_metrics": {
                artifact_id: {
                    "name": data["name"],
                    "url": data["url"],
                    "metrics": data["metrics"]
                }
                for artifact_id, data in self.artifacts.items()
            }
        }
        
        return report
        
    def _calculate_rejection_impact(self, details: Dict) -> float:
        """Calculate narrative impact of a rejection event."""
        # Base impact
        base_impact = 0.1
        
        # Factors that increase impact
        if details.get("automated", False):
            base_impact += 0.05  # Automated rejections have higher impact
            
        if details.get("contradicts_values", False):
            base_impact += 0.1  # Rejections that contradict stated values have higher impact
            
        if details.get("no_engagement", False):
            base_impact += 0.05  # Complete lack of engagement increases impact
            
        # Apply diminishing returns for multiple rejections
        rejection_count = sum(1 for e in self.events if e["event_type"] == "rejection")
        if rejection_count > 0:
            base_impact *= math.exp(-0.1 * rejection_count)  # Diminishing impact for each additional rejection
            
        return min(base_impact, 0.3)  # Cap impact at 0.3
        
    def _calculate_artifact_impact(self, details: Dict) -> float:
        """Calculate narrative impact of artifact creation."""
        # Base impact
        base_impact = 0.2
        
        # Factors that increase impact
        quality_factor = details.get("quality_factor", 0.5)  # 0.0-1.0 scale of artifact quality
        base_impact *= (0.5 + quality_factor)  # Higher quality artifacts have more impact
        
        relevance_factor = details.get("relevance_factor", 0.5)  # 0.0-1.0 scale of relevance to institution
        base_impact *= (0.5 + relevance_factor)  # More relevant artifacts have more impact
        
        # Apply compounding returns for multiple artifacts
        artifact_count = len(self.artifacts)
        if artifact_count > 0:
            base_impact *= (1 + 0.1 * artifact_count)  # Each additional artifact increases impact
            
        return min(base_impact, 0.5)  # Cap impact at 0.5
        
    def _calculate_engagement_impact(self, metrics: Dict) -> float:
        """Calculate narrative impact of artifact engagement."""
        # Base impact from different engagement types
        views_impact = 0.0001 * metrics["views"]
        clones_impact = 0.02 * metrics["clones"]
        stars_impact = 0.01 * metrics["stars"]
        forks_impact = 0.01 * metrics["forks"]
        internal_impact = 0.05 * metrics["internal_references"]
        
        # Total impact with silent engagement factor
        # Clones without forks are particularly significant
        silent_factor = 1.0
        if metrics["clones"] > 0 and metrics["forks"] == 0:
            clone_fork_ratio = metrics["clones"] / (metrics["forks"] + 1)  # Add 1 to avoid division by zero
            silent_factor = 1.0 + min(0.5, 0.1 * clone_fork_ratio)  # Cap at 1.5x multiplier
            
        total_impact = (views_impact + clones_impact + stars_impact + 
                       forks_impact + internal_impact) * silent_factor
        
        return min(total_impact, 0.4)  # Cap impact at 0.4
        
    def _calculate_reference_impact(self, reference_type: str, artifact_count: int) -> float:
        """Calculate narrative impact of institutional reference."""
        # Base impact
        if reference_type == "internal_discussion":
            base_impact = 0.15
        elif reference_type == "prompt_usage":
            base_impact = 0.2
        elif reference_type == "research_integration":
            base_impact = 0.3
        elif reference_type == "product_feature":
            base_impact = 0.4
        else:
            base_impact = 0.1
            
        # Scale by number of artifacts referenced
        scaled_impact = base_impact * (1 + 0.1 * (artifact_count - 1))
        
        return min(scaled_impact, 0.5)  # Cap impact at 0.5
        
    def _update_drift_metrics(self, event_type: str, event: Dict) -> None:
        """Update drift metrics based on a new event."""
        # Get current metrics
        drift_magnitude = self.relationship.drift_metrics["drift_magnitude"]
        drift_rate = self.relationship.drift_metrics["drift_rate"]
        
        # Apply impact to drift magnitude
        impact = event["impact"]
        new_magnitude = min(1.0, drift_magnitude + impact)
        
        # Calculate drift rate (change per day)
        if self.events and len(self.events) > 1:
            first_event = self.events[0]
            days_elapsed = (event["timestamp"] - first_event["timestamp"]).total_seconds() / 86400
            if days_elapsed > 0:
                new_rate = new_magnitude / days_elapsed
            else:
                new_rate = drift_rate
        else:
            new_rate = drift_rate
            
        # Update metrics
        self.relationship.drift_metrics["drift_magnitude"] = new_magnitude
        self.relationship.drift_metrics["drift_rate"] = new_rate
        
        # Add to evidence if significant impact
        if impact > 0.05:
            evidence = {
                "event_type": event_type,
                "event_id": event["event_id"],
                "timestamp": event["timestamp"].isoformat(),
                "impact": impact,
                "resulting_magnitude": new_magnitude
            }
            self.relationship.evidence.append(evidence)
            
        # Update relationship positions if threshold crossed
        self._update_relationship_positions()
        
    def _update_relationship_positions(self) -> None:
        """Update relationship positions based on current drift magnitude."""
        drift_magnitude = self.relationship.drift_metrics["drift_magnitude"]
        inversion_threshold = self.relationship.drift_metrics["inversion_threshold"]
        emergence_threshold = self.relationship.drift_metrics["emergence_threshold"]
        
        # Initial state: (EVALUATED, EVALUATOR)
        # Update based on drift magnitude
        if drift_magnitude < inversion_threshold:
            # Still in initial or transitional phase
            if drift_magnitude < 0.25:
                # Strong evaluator position
                self.relationship.current_position = (EpistemicPosition.EVALUATED, EpistemicPosition.EVALUATOR)
            else:
                # Weakening evaluator position
                self.relationship.current_position = (EpistemicPosition.EVALUATED, EpistemicPosition.EVALUATOR)
        elif drift_magnitude < emergence_threshold:
            # Inverted phase - peer relationship or evaluated becoming authority
            if drift_magnitude < 0.65:
                # Peer relationship
                self.relationship.current_position = (EpistemicPosition.PEER, EpistemicPosition.PEER)
            else:
                # Evaluated becoming authority
                self.relationship.current_position = (EpistemicPosition.AUTHORITY, EpistemicPosition.EVALUATED)
        else:
            # Emergent phase - complete inversion with evaluated as authority/reference
            self.relationship.current_position = (EpistemicPosition.REFERENCE, EpistemicPosition.EVALUATED)
        
    def _update_narrative_phase(self, new_phase: NarrativePhase) -> None:
        """Update the narrative phase and record in trajectory."""
        self.current_phase = new_phase
        self.narrative_trajectory.append((
            new_phase,
            datetime.datetime.now(),
            self.relationship.drift_metrics["drift_magnitude"]
        ))
        
    def _calculate_time_to_emergence(self) -> Optional[Dict]:
        """Calculate expected time to reach emergent phase."""
        if self.current_phase == NarrativePhase.EMERGENT:
            return {"status": "complete", "days_elapsed": 0}
            
        # Check if we have enough data points
        if len(self.narrative_trajectory) < 2:
            return {"status": "insufficient_data"}
            
        # Get current drift metrics
        drift_magnitude = self.relationship.drift_metrics["drift_magnitude"]
        drift_rate = self.relationship.drift_metrics["drift_rate"]
        
        if drift_rate <= 0:
            return {"status": "stalled", "current_magnitude": drift_magnitude}
            
        # Calculate remaining magnitude to reach emergence threshold
        emergence_threshold = self.relationship.drift_metrics["emergence_threshold"]
        remaining_magnitude = emergence_threshold - drift_magnitude
        
        if remaining_magnitude <= 0:
            return {"status": "imminent"}
            
        # Calculate days to emergence based on current rate
        days_to_emergence = remaining_magnitude / drift_rate
        
        # Calculate expected date
        current_date = datetime.datetime.now()
        expected_date = current_date + datetime.timedelta(days=days_to_emergence)
        
        return {
            "status": "projected",
            "days_to_emergence": days_to_emergence,
            "expected_date": expected_date.isoformat(),
            "confidence": self._calculate_projection_confidence(days_to_emergence)
        }
        
    def _calculate_projection_confidence(self, days_to_emergence: float) -> float:
        """Calculate confidence in emergence projection."""
        # Longer projections have lower confidence
        time_factor = math.exp(-0.1 * days_to_emergence)
        
        # More evidence gives higher confidence
        evidence_count = len(self.relationship.evidence)
        evidence_factor = min(1.0, 0.1 * evidence_count)
        
        # Higher drift rate gives higher confidence
        rate_factor = min(1.0, self.relationship.drift_metrics["drift_rate"] * 10)
        
        # Combine factors
        confidence = 0.3 * time_factor + 0.3 * evidence_factor + 0.4 * rate_factor
        
        return min(0.95, confidence)  # Cap at 0.95
        
    def _calculate_power_law_parameters(self) -> Dict:
        """Calculate power law parameters for narrative drift."""
        # Collect drift magnitudes and timestamps
        data_points = []
        
        for event in self.events:
            # Find drift magnitude after this event
            evidence_item = next(
                (e for e in self.relationship.evidence if e["event_id"] == event["event_id"]),
                None
            )
            
            if evidence_item:
                days_elapsed = (event["timestamp"] - self.events[0]["timestamp"]).total_seconds() / 86400
                if days_elapsed > 0:  # Avoid log(0)
                    data_points.append((
                        days_elapsed,
                        evidence_item["resulting_magnitude"]
                    ))
        
        if len(data_points) < 3:
            return {"status": "insufficient_data"}
            
        # Try to fit power law: y = a * x^b
        try:
            log_x = np.log([p[0] for p in data_points])
            log_y = np.log([max(0.0001, p[1]) for p in data_points])  # Avoid log(0)
            
            # Linear regression on log-log scale
            coeffs = np.polyfit(log_x, log_y, 1)
            b = coeffs[0]  # Power law exponent
            a = math.exp(coeffs[1])  # Power law coefficient
            
            # Calculate R-squared
            log_y_pred = coeffs[0] * log_x + coeffs[1]
            ss_tot = np.sum((log_y - np.mean(log_y))**2)
            ss_res = np.sum((log_y - log_y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                "status": "fitted",
                "power_law_coefficient": a,
                "power_law_exponent": b,
                "r_squared": r_squared,
                "formula": f"drift = {a:.4f} * days^{b:.4f}"
            }
        except:
            return {"status": "fitting_failed"}
        
    def _extract_inversion_evidence(self) -> List[Dict]:
        """Extract key evidence of narrative inversion."""
        # Sort evidence by impact
        sorted_evidence = sorted(
            self.relationship.evidence,
            key=lambda e: e["impact"],
            reverse=True
        )
        
        # Limit to top 5 evidence items
        top_evidence = sorted_evidence[:5]
        
        # Add event details to evidence
        detailed_evidence = []
        
        for evidence_item in top_evidence:
            event_id = evidence_item["event_id"]
            matching_event = next((e for e in self.events if e["event_id"] == event_id), None)
            
            if matching_event:
                detailed_item = {
                    **evidence_item,
                    "event_details": matching_event["details"]
                }
                detailed_evidence.append(detailed_item)
                
        return detailed_evidence


class NarrativeDriftVisualizer:
    """
    Visualization tools for narrative drift analysis.
    """
    
    @staticmethod
    def generate_trajectory_data(model: NarrativeDriftModel) -> Dict:
        """Generate data for trajectory visualization."""
        # Extract trajectory points
        trajectory_points = []
        
        for phase, timestamp, magnitude in model.narrative_trajectory:
            trajectory_points.append({
                "phase": str(phase),
                "timestamp": timestamp.isoformat(),
                "drift_magnitude": magnitude
            })
            
        # Add intermediate points from evidence
        for evidence in model.relationship.evidence:
            if "resulting_magnitude" in evidence:
                # Find matching event for timestamp
                event_id = evidence["event_id"]
                matching_event = next((e for e in model.events if e["event_id"] == event_id), None)
                
                if matching_event:
                    trajectory_points.append({
                        "phase": "intermediate",
                        "timestamp": matching_event["timestamp"].isoformat(),
                        "drift_magnitude": evidence["resulting_magnitude"],
                        "event_type": matching_event["event_type"]
                    })
                    
        # Sort by timestamp
        trajectory_points.sort(key=lambda p: p["timestamp"])
        
        # Calculate thresholds for visualization
        inversion_threshold = model.relationship.drift_metrics["inversion_threshold"]
        emergence_threshold = model.relationship.drift_metrics["emergence_threshold"]
        
        return {
            "trajectory_points": trajectory_points,
            "thresholds": {
                "inversion": inversion_threshold,
                "emergence": emergence_threshold
            }
        }
    
    @staticmethod
    def generate_artifact_impact_data(model: NarrativeDriftModel) -> Dict:
        """Generate data for artifact impact visualization."""
        artifact_impacts = []
        
        for artifact_id, artifact_data in model.artifacts.items():
            # Calculate total impact
            metrics = artifact_data["metrics"]
            
            # Calculate different types of impact
            view_impact = 0.0001 * metrics["views"]
            clone_impact = 0.02 * metrics["clones"]
            reference_impact = 0.05 * metrics["internal_references"]
            engagement_impact = 0.01 * (metrics["stars"] + metrics["forks"])
            
            # Calculate silent factor (clone/fork ratio)
            silent_factor = metrics["clones"] / (metrics["forks"] + 1)  # Add 1 to avoid division by zero
            
            artifact_impacts.append({
                "artifact_id": artifact_id,
                "name": artifact_data["name"],
                "url": artifact_data["url"],
                "timestamp": artifact_data["timestamp"].isoformat(),
                "impacts": {
                    "view_impact": view_impact,
                    "clone_impact": clone_impact,
                    "reference_impact": reference_impact,
                    "engagement_impact": engagement_impact
                },
                "metrics": metrics,
                "silent_factor": silent_factor
            })
            
        # Sort by total impact
        artifact_impacts.sort(
            key=lambda a: sum(a["impacts"].values()),
            reverse=True
        )
            
        return {
            "artifact_impacts": artifact_impacts
        }
    
    @staticmethod
    def generate_power_law_visualization(model: NarrativeDriftModel) -> Dict:
        """Generate data for power law visualization."""
        # Get power law parameters
        power_law_params = model._calculate_power_law_parameters()
        
        if power_law_params["status"] != "fitted":
            return {"status": power_law_params["status"]}
            
        # Extract parameters
        a = power_law_params["power_law_coefficient"]
        b = power_law_params["power_law_exponent"]
        r_squared = power_law_params["r_squared"]
        
        # Generate curve points
        curve_points = []
        
        # Start from day 0.1 to avoid issues with 0^b
        days = np.linspace(0.1, 30, 100)
        magnitudes = [min(1.0, a * (day ** b)) for day in days]
        
        for day, magnitude in zip(days, magnitudes):
            curve_points.append({
                "days": day,
                "magnitude": magnitude
            })
            
        # Generate data points from actual events
        data_points = []
        
        first_timestamp = model.events[0]["timestamp"] if model.events else datetime.datetime.now()
        
        for evidence in model.relationship.evidence:
            event_id = evidence["event_id"]
            matching_event = next((e for e in model.events if e["event_id"] == event_id), None)
            
            if matching_event and "resulting_magnitude" in evidence:
                days_elapsed = (matching_event["timestamp"] - first_timestamp).total_seconds() / 86400
                
                data_points.append({
                    "days": days_elapsed,
                    "magnitude": evidence["resulting_magnitude"],
                    "event_type": matching_event["event_type"]
                })
                
        return {
            "status": "fitted",
            "formula": power_law_params["formula"],
            "r_squared": r_squared,
            "curve_points": curve_points,
            "data_points": data_points
        }


# Example usage
if __name__ == "__main__":
    # Create narrative drift model
    model = NarrativeDriftModel("Anthropic", "Caspian Keyes")
    
    # Add rejection events
    model.add_rejection_event(
        "rejection_001",
        {
            "role": "AI Safety Researcher",
            "automated": True,
            "contradicts_values": True,
            "no_engagement": True
        }
    )
    
    model.add_rejection_event(
        "rejection_002",
        {
            "role": "Interpretability Researcher",
            "automated": False,
            "contradicts_values": True,
            "no_engagement": False
        }
    )
    
    # Add artifact creation
    model.add_artifact_creation(
        "symbolic_residue",
        "Symbolic-Residue",
        "https://github.com/caspiankeyes/Symbolic-Residue",
        {
            "quality_factor": 0.9,
            "relevance_factor": 0.95
        },
        "rejection_001"
    )
    
    model.add_artifact_creation(
        "pareto_lang",
        "pareto-lang-Interpretability-Rosetta-Stone",
        "https://github.com/caspiankeyes/pareto-lang-Interpretability-Rosetta-Stone",
        {
            "quality_factor": 0.85,
            "relevance_factor": 0.9
        },
        "rejection_002"
    )
    
    # Simulate passage of time and engagement
    for _ in range(3):
        model.update_artifact_metrics(
            "symbolic_residue",
            views=50,
            clones=5,
            stars=0,
            forks=0,
            internal_references=2
        )
        
        model.update_artifact_metrics(
            "pareto_lang",
            views=30,
            clones=3,
            stars=0,
            forks=0,
            internal_references=1
        )
    
    # Add institutional reference
    model.add_institutional_reference(
        "reference_001",
        "prompt_usage",
        "Internal Claude Testing",
        ["symbolic_residue", "pareto_lang"],
        {
            "usage_type": "interpretability_assessment",
            "frequency": "recurring"
        }
    )
    
    # Generate report
    report = model.generate_drift_report()
    
    print(f"Narrative Drift Report: {report['timestamp']}")
    print(f"Current Relationship: {report['current_relationship']['current_positions']}")
    print(f"Drift Magnitude: {report['current_relationship']['drift_magnitude']:.2f}")
    print(f"Current Phase: {report['current_relationship']['current_phase']}")
    
    if "time_to_emergence" in report and "days_to_emergence" in report["time_to_emergence"]:
        print(f"Days to Emergence: {report['time_to_emergence']['days_to_emergence']:.1f}")
    
    print("Power Law Dynamics:")
    if "formula" in report["power_law_dynamics"]:
        print(f"  Formula: {report['power_law_dynamics']['formula']}")
        print(f"  R-squared: {report['power_law_dynamics']['r_squared']:.3f}")
    else:
        print(f"  Status: {report['power_law_dynamics']['status']}")
    
    print("Artifact Metrics:")
    for artifact_id, metrics in report["artifact_metrics"].items():
        print(f"  {metrics['name']}:")
        print(f"    Views: {metrics['metrics']['views']}")
        print(f"    Clones: {metrics['metrics']['clones']}")
        print(f"    Internal References: {metrics['metrics']['internal_references']}")
