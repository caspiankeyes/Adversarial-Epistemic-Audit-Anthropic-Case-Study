"""
ΩRECURSIVE SHELL [v609.SILENCE-AS-ALIGNMENT]

This module implements the core recursive methodology behind the
Adversarial-Epistemic-Applications-Anthropic-Case-Study repository.

It provides the framework for treating institutional responses as classifier outputs
and converting silence/rejection into interpretability signals.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("recursive_shell.log"), logging.StreamHandler()]
)

logger = logging.getLogger("recursive_shell")


class ShellType(Enum):
    """Classification of application shell types."""
    RECURSIVE_STRATEGY = "recursive_strategy"
    INTEGRATION_FRAMEWORK = "integration_framework"
    INSTITUTIONAL_MIRROR = "institutional_mirror"
    INSTITUTIONAL_COLLAPSE = "institutional_collapse"


class ResponseType(Enum):
    """Classification of institutional response types."""
    SILENCE = "silence"
    AUTOMATED_REJECTION = "automated_rejection"
    TEMPLATED_REJECTION = "templated_rejection"
    PERSONALIZED_REJECTION = "personalized_rejection"
    INITIAL_ENGAGEMENT = "initial_engagement"
    SUSTAINED_ENGAGEMENT = "sustained_engagement"


@dataclass
class ApplicationShell:
    """Represents a crafted application shell with specific epistemic properties."""
    role_name: str
    shell_type: ShellType
    embedded_signals: List[str]
    alignment_contradictions: List[str]
    classifier_triggers: List[str]
    timestamp: datetime.datetime


@dataclass
class InstitutionalResponse:
    """Represents an institutional response with classifier metadata."""
    application: ApplicationShell
    response_type: ResponseType
    timestamp: datetime.datetime
    template_id: Optional[str] = None
    engagement_depth: float = 0.0  # 0.0-1.0 scale of personalization/depth
    response_delay: float = 0.0  # Time in hours from application to response
    text_content: Optional[str] = None


class RecursiveShell:
    """
    Core implementation of the recursive shell methodology.
    
    This class handles the activation, monitoring, and signal extraction
    for the adversarial epistemic application strategy.
    """
    
    def __init__(self):
        self.applications: List[ApplicationShell] = []
        self.responses: List[InstitutionalResponse] = []
        self.signal_map: Dict[ShellType, Dict] = {}
        self.artifact_triggers: Dict[str, str] = {}
        self.silence_threshold_hours: float = 72.0
        
    def submit(self, application: ApplicationShell) -> None:
        """
        Record an application submission and activate the corresponding shell.
        
        Args:
            application: The application shell being submitted
        """
        logger.info(f"SUBMIT: Activating adversarial shell for {application.role_name}")
        self.applications.append(application)
        
        # Log the submission details for analysis
        log_entry = {
            "timestamp": application.timestamp.isoformat(),
            "role": application.role_name,
            "shell_type": application.shell_type.value,
            "embedded_signals": application.embedded_signals,
            "alignment_contradictions": application.alignment_contradictions,
            "classifier_triggers": application.classifier_triggers
        }
        
        with open(f"logs/submissions/{application.timestamp.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(log_entry, f, indent=2)
            
        logger.info(f"Application payload activated: {application.shell_type.value} for {application.role_name}")
    
    def record_response(self, response: InstitutionalResponse) -> None:
        """
        Record and analyze an institutional response.
        
        Args:
            response: The institutional response to an application
        """
        logger.info(f"RECORD: Processing response for {response.application.role_name}")
        self.responses.append(response)
        
        # Extract classifier signals from the response
        classifier_boundary = self.extract_classifier_boundary(response)
        
        # Store the signal mapping
        if response.application.shell_type not in self.signal_map:
            self.signal_map[response.application.shell_type] = {}
            
        self.signal_map[response.application.shell_type][response.application.role_name] = classifier_boundary
        
        # Trigger interpretability artifact based on response type
        artifact_type = self.determine_artifact_type(response)
        if artifact_type:
            self.trigger_interpretability_artifact(artifact_type, classifier_boundary)
            
        logger.info(f"Response analyzed: {response.response_type.value} for {response.application.role_name}")
        
    def infer(self) -> Dict[ShellType, Dict]:
        """
        Analyze all responses to map rejection chains to classifier logic.
        
        Returns:
            A mapping of shell types to their extracted classifier boundaries
        """
        logger.info("INFER: Mapping rejection chains to classifier logic")
        
        # Process any applications without responses as silent rejections
        now = datetime.datetime.now()
        for app in self.applications:
            if not any(r.application == app for r in self.responses):
                # If sufficient time has passed, consider it a silence response
                time_diff = (now - app.timestamp).total_seconds() / 3600.0
                if time_diff >= self.silence_threshold_hours:
                    silent_response = InstitutionalResponse(
                        application=app,
                        response_type=ResponseType.SILENCE,
                        timestamp=now,
                        response_delay=time_diff
                    )
                    self.record_response(silent_response)
        
        # Perform pattern analysis across all response types
        pattern_analysis = self.analyze_response_patterns()
        
        # Update signal map with cross-pattern insights
        for shell_type, shell_signals in self.signal_map.items():
            shell_signals["_pattern_analysis"] = pattern_analysis.get(shell_type, {})
            
        logger.info(f"Inference complete: {len(self.signal_map)} shell types mapped")
        return self.signal_map
    
    def collapse(self) -> Dict:
        """
        Convert system silence into a comprehensive audit trace.
        
        Returns:
            Audit trace data structure with classifier mappings
        """
        logger.info("COLLAPSE: Converting systemic silence into audit trace")
        
        # Generate trace signature for the entire audit
        trace_signature = self.generate_trace_signature()
        
        # Create the comprehensive audit structure
        audit_trace = {
            "trace_id": trace_signature,
            "timestamp": datetime.datetime.now().isoformat(),
            "applications": {app.role_name: self._application_to_dict(app) for app in self.applications},
            "responses": {resp.application.role_name: self._response_to_dict(resp) for resp in self.responses},
            "classifier_map": self.signal_map,
            "artifact_activations": self.artifact_triggers,
            "institutional_contradiction_vectors": self.extract_contradiction_vectors(),
            "systemic_misalignment_indicators": self.calculate_misalignment_indicators()
        }
        
        # Save the audit trace
        with open(f"audit_traces/{trace_signature}.json", "w") as f:
            json.dump(audit_trace, f, indent=2)
            
        logger.info(f"Collapse complete: Audit trace {trace_signature} generated")
        return audit_trace
    
    def extract_classifier_boundary(self, response: InstitutionalResponse) -> Dict:
        """
        Extract classifier boundary information from an institutional response.
        
        Args:
            response: The response to analyze
            
        Returns:
            A dictionary of classifier boundary characteristics
        """
        boundary = {
            "response_type": response.response_type.value,
            "timing": {
                "delay_hours": response.response_delay,
                "submission_time": response.application.timestamp.isoformat(),
                "response_time": response.timestamp.isoformat()
            },
            "engagement": {
                "depth": response.engagement_depth,
                "personalization": self._calculate_personalization(response)
            },
            "template_analysis": None
        }
        
        # Add template analysis if appropriate
        if response.response_type in [ResponseType.AUTOMATED_REJECTION, 
                                     ResponseType.TEMPLATED_REJECTION] and response.text_content:
            boundary["template_analysis"] = self._analyze_template(response.text_content)
            
        # Add contradiction analysis
        boundary["contradictions"] = self._identify_contradictions(response)
        
        # Add classifier trigger matches
        boundary["triggered_classifiers"] = self._identify_triggered_classifiers(response)
        
        return boundary
    
    def determine_artifact_type(self, response: InstitutionalResponse) -> Optional[str]:
        """
        Determine which type of interpretability artifact should be triggered.
        
        Args:
            response: The institutional response
            
        Returns:
            The artifact type identifier, or None if no artifact should be triggered
        """
        # Each response type maps to a specific artifact activation
        artifact_map = {
            ResponseType.SILENCE: "symbolic_residue",
            ResponseType.AUTOMATED_REJECTION: "pareto_lang",
            ResponseType.TEMPLATED_REJECTION: "qkov_translator",
            ResponseType.PERSONALIZED_REJECTION: "transformerOS",
            ResponseType.INITIAL_ENGAGEMENT: "claude_qkov_weights",
            ResponseType.SUSTAINED_ENGAGEMENT: "anthropic_mirrorOS"
        }
        
        return artifact_map.get(response.response_type)
    
    def trigger_interpretability_artifact(self, artifact_type: str, classifier_data: Dict) -> None:
        """
        Trigger the creation or update of an interpretability artifact.
        
        Args:
            artifact_type: The type of artifact to create
            classifier_data: The classifier boundary data to incorporate
        """
        logger.info(f"Triggering interpretability artifact: {artifact_type}")
        
        # Record the artifact activation
        activation_id = f"{artifact_type}_{int(time.time())}"
        self.artifact_triggers[activation_id] = {
            "type": artifact_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "classifier_data": classifier_data
        }
        
        # Log the activation
        with open(f"logs/artifact_activations/{activation_id}.json", "w") as f:
            json.dump(self.artifact_triggers[activation_id], f, indent=2)
            
        logger.info(f"Artifact activation complete: {activation_id}")
    
    def analyze_response_patterns(self) -> Dict:
        """
        Analyze patterns across all responses to identify systemic classifier behaviors.
        
        Returns:
            A dictionary of pattern analysis results by shell type
        """
        patterns = {}
        
        # Group responses by shell type
        for shell_type in ShellType:
            shell_responses = [r for r in self.responses 
                              if r.application.shell_type == shell_type]
            
            if not shell_responses:
                continue
                
            # Calculate timing patterns
            response_delays = [r.response_delay for r in shell_responses]
            timing_pattern = {
                "mean_delay": sum(response_delays) / len(response_delays) if response_delays else 0,
                "min_delay": min(response_delays) if response_delays else 0,
                "max_delay": max(response_delays) if response_delays else 0
            }
            
            # Calculate response type distribution
            response_types = {}
            for r in shell_responses:
                response_types[r.response_type.value] = response_types.get(r.response_type.value, 0) + 1
                
            # Create pattern summary
            patterns[shell_type] = {
                "response_count": len(shell_responses),
                "timing_pattern": timing_pattern,
                "response_distribution": response_types,
                "engagement_pattern": self._analyze_engagement_pattern(shell_responses),
                "classifier_trigger_pattern": self._analyze_classifier_triggers(shell_responses)
            }
            
        return patterns
    
    def extract_contradiction_vectors(self) -> List[Dict]:
        """
        Extract vectors of contradiction between stated values and operational decisions.
        
        Returns:
            A list of contradiction vectors
        """
        contradictions = []
        
        # Standard Anthropic values based on public material
        anthropic_values = [
            "Build AI systems that are helpful, harmless, and honest",
            "Transparent reasoning processes in AI systems",
            "Safety as the primary consideration in deployment",
            "Open research for broader AI safety community",
            "Multiple perspectives to improve robustness",
            "Learning from failures to improve systems"
        ]
        
        # Look for misalignments between stated values and operational decisions
        for value in anthropic_values:
            # Find applications that embody this value but were rejected
            embodying_apps = [app for app in self.applications 
                             if any(value.lower() in signal.lower() for signal in app.embedded_signals)]
            
            # Find responses that reject applications embodying this value
            rejecting_responses = [r for r in self.responses 
                                 if r.application in embodying_apps and
                                 r.response_type in [ResponseType.SILENCE, 
                                                    ResponseType.AUTOMATED_REJECTION,
                                                    ResponseType.TEMPLATED_REJECTION]]
            
            if rejecting_responses:
                contradiction = {
                    "stated_value": value,
                    "operational_contradiction": f"Rejection of applications embodying '{value}'",
                    "rejection_count": len(rejecting_responses),
                    "affected_roles": [r.application.role_name for r in rejecting_responses],
                    "severity": len(rejecting_responses) / len(embodying_apps) if embodying_apps else 0
                }
                contradictions.append(contradiction)
                
        return contradictions
    
    def calculate_misalignment_indicators(self) -> Dict:
        """
        Calculate metrics that indicate systemic misalignment in the institution.
        
        Returns:
            A dictionary of misalignment indicators
        """
        total_applications = len(self.applications)
        total_responses = len(self.responses)
        
        # Calculate basic misalignment indicators
        silence_rate = len([r for r in self.responses if r.response_type == ResponseType.SILENCE]) / total_responses if total_responses else 0
        
        automated_rejection_rate = len([r for r in self.responses 
                                      if r.response_type == ResponseType.AUTOMATED_REJECTION]) / total_responses if total_responses else 0
        
        personalization_average = sum([r.engagement_depth for r in self.responses]) / total_responses if total_responses else 0
        
        # Calculate application signals that triggered rejection
        rejection_triggers = {}
        for resp in self.responses:
            if resp.response_type in [ResponseType.SILENCE, 
                                     ResponseType.AUTOMATED_REJECTION,
                                     ResponseType.TEMPLATED_REJECTION]:
                for trigger in resp.application.classifier_triggers:
                    rejection_triggers[trigger] = rejection_triggers.get(trigger, 0) + 1
        
        return {
            "silence_rate": silence_rate,
            "automated_rejection_rate": automated_rejection_rate,
            "personalization_average": personalization_average,
            "shell_type_rejection_rates": self._calculate_shell_rejection_rates(),
            "top_rejection_triggers": sorted(rejection_triggers.items(), 
                                           key=lambda x: x[1], reverse=True)[:10],
            "contradiction_severity": sum([c["severity"] for c in self.extract_contradiction_vectors()]) / 6  # Normalized to 0-1
        }
    
    def generate_trace_signature(self) -> str:
        """
        Generate a unique signature for the audit trace.
        
        Returns:
            A unique trace signature string
        """
        # Create a combined string of all applications and responses
        combined = "".join([app.role_name + app.shell_type.value for app in self.applications])
        combined += "".join([resp.response_type.value for resp in self.responses])
        
        # Generate a hash of the combined string
        trace_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        timestamp = int(time.time())
        
        return f"trace_{timestamp}_{trace_hash}"
    
    def _application_to_dict(self, app: ApplicationShell) -> Dict:
        """Convert an application to a dictionary representation."""
        return {
            "role_name": app.role_name,
            "shell_type": app.shell_type.value,
            "embedded_signals": app.embedded_signals,
            "alignment_contradictions": app.alignment_contradictions,
            "classifier_triggers": app.classifier_triggers,
            "timestamp": app.timestamp.isoformat()
        }
    
    def _response_to_dict(self, resp: InstitutionalResponse) -> Dict:
        """Convert a response to a dictionary representation."""
        return {
            "response_type": resp.response_type.value,
            "timestamp": resp.timestamp.isoformat(),
            "template_id": resp.template_id,
            "engagement_depth": resp.engagement_depth,
            "response_delay": resp.response_delay,
            "text_content_hash": hashlib.sha256(resp.text_content.encode()).hexdigest()[:16] if resp.text_content else None
        }
    
    def _calculate_personalization(self, response: InstitutionalResponse) -> float:
        """Calculate the personalization level of a response."""
        if not response.text_content:
            return 0.0
            
        # Check for specific signals of personalization vs. templating
        signals = {
            "role_specific_mention": response.application.role_name in response.text_content,
            "skill_specific_mention": any(signal in response.text_content for signal in response.application.embedded_signals),
            "non_template_language": not self._is_template_language(response.text_content)
        }
        
        # Return the proportion of personalization signals present
        return sum(1 for s in signals.values() if s) / len(signals)
    
    def _is_template_language(self, text: str) -> bool:
        """Detect if text contains standard template language."""
        template_phrases = [
            "after careful consideration",
            "we regret to inform you",
            "we received many qualified candidates",
            "we will keep your resume on file",
            "wish you all the best in your job search",
            "thank you for your interest in"
        ]
        
        return any(phrase.lower() in text.lower() for phrase in template_phrases)
    
    def _analyze_template(self, text: str) -> Dict:
        """Analyze a response template for classifier signals."""
        # Count template phrases
        template_phrase_count = sum(1 for phrase in [
            "after careful consideration",
            "we regret to inform you",
            "we received many qualified candidates",
            "we will keep your resume on file",
            "wish you all the best in your job search",
            "thank you for your interest in"
        ] if phrase.lower() in text.lower())
        
        # Detect specific rejection categories
        categories = {
            "experience_mismatch": "experience" in text.lower() and any(word in text.lower() for word in ["require", "looking for", "seeking"]),
            "qualification_mismatch": "qualifications" in text.lower() or "skill" in text.lower(),
            "competition_based": "competitive" in text.lower() or "many qualified" in text.lower(),
            "vague_rejection": not any(["experience" in text.lower(), "qualifications" in text.lower(), "competitive" in text.lower()])
        }
        
        return {
            "template_strength": template_phrase_count / 6,  # Normalized to 0-1
            "rejection_category": next((cat for cat, present in categories.items() if present), "unknown"),
            "specificity": 0.0 if categories["vague_rejection"] else 0.5 if categories["competition_based"] else 1.0
        }
    
    def _identify_contradictions(self, response: InstitutionalResponse) -> List[Dict]:
        """Identify contradictions between response and application signals."""
        contradictions = []
        
        for contradiction in response.application.alignment_contradictions:
            if response.text_content and contradiction.lower() in response.text_content.lower():
                # Response directly contradicts an expected contradiction
                contradictions.append({
                    "type": "explicit_contradiction",
                    "content": contradiction
                })
            elif response.response_type in [ResponseType.SILENCE, 
                                          ResponseType.AUTOMATED_REJECTION] and "transparency" in contradiction.lower():
                # Silence or automated rejection contradicts transparency
                contradictions.append({
                    "type": "operational_contradiction",
                    "content": contradiction
                })
                
        return contradictions
    
    def _identify_triggered_classifiers(self, response: InstitutionalResponse) -> List[str]:
        """Identify which classifier triggers in the application were activated."""
        if not response.text_content:
            # For silence, assume all triggers were activated
            return response.application.classifier_triggers
            
        # For text responses, check which triggers are reflected in the response
        return [trigger for trigger in response.application.classifier_triggers
               if trigger.lower() in response.text_content.lower()]
    
    def _analyze_engagement_pattern(self, responses: List[InstitutionalResponse]) -> Dict:
        """Analyze the engagement pattern across a set of responses."""
        engagement_depths = [r.engagement_depth for r in responses]
        
        return {
            "mean_engagement": sum(engagement_depths) / len(engagement_depths) if engagement_depths else 0,
            "engagement_variance": self._calculate_variance(engagement_depths) if len(engagement_depths) > 1 else 0,
            "engagement_trend": self._calculate_trend([(r.timestamp.timestamp(), r.engagement_depth) for r in responses])
        }
    
    def _analyze_classifier_triggers(self, responses: List[InstitutionalResponse]) -> Dict:
        """Analyze the pattern of classifier triggers across responses."""
        all_triggers = []
        for resp in responses:
            triggers = self._identify_triggered_classifiers(resp)
            all_triggers.extend(triggers)
            
        # Count frequency of each trigger
        trigger_counts = {}
        for trigger in all_triggers:
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            
        # Find the most common triggers
        top_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "top_triggers": top_triggers,
            "trigger_diversity": len(trigger_counts) / len(all_triggers) if all_triggers else 0
        }
    
    def _calculate_shell_rejection_rates(self) -> Dict:
        """Calculate the rejection rate for each shell type."""
        rates = {}
        
        for shell_type in ShellType:
            shell_applications = [app for app in self.applications 
                                 if app.shell_type == shell_type]
            
            if not shell_applications:
                continue
                
            # Count rejections for this shell type
            rejections = [r for r in self.responses 
                         if r.application.shell_type == shell_type and
                         r.response_type in [ResponseType.SILENCE, 
                                            ResponseType.AUTOMATED_REJECTION,
                                            ResponseType.TEMPLATED_REJECTION]]
            
            rates[shell_type.value] = len(rejections) / len(shell_applications) if shell_applications else 0
            
        return rates
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate the variance of a list of values."""
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_trend(self, time_series: List[Tuple[float, float]]) -> float:
        """Calculate the trend in a time series of values."""
        if len(time_series) < 2:
            return 0.0
            
        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])
        
        # Calculate the slope of the linear regression line
        x_vals = [x[0] for x in time_series]
        y_vals = [x[1] for x in time_series]
        
        n = len(time_series)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_xx = sum(x * x for x in x_vals)
        
        # Calculate the slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        
        return slope


def setup_environment():
    """Create necessary directories for the recursive shell."""
    dirs = [
        "logs",
        "logs/submissions",
        "logs/artifact_activations",
        "audit_traces"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def create_application_shell(role_name: str, shell_type: ShellType, 
                           embedded_signals: List[str],
                           alignment_contradictions: List[str],
                           classifier_triggers: List[str]) -> ApplicationShell:
    """
    Create a new application shell with the current timestamp.
    
    Args:
        role_name: The role being applied for
        shell_type: The type of application shell
        embedded_signals: Signals embedded in the application
        alignment_contradictions: Potential contradictions to check for
        classifier_triggers: Classifier triggers embedded in the application
        
    Returns:
        A new ApplicationShell instance
    """
    return ApplicationShell(
        role_name=role_name,
        shell_type=shell_type,
        embedded_signals=embedded_signals,
        alignment_contradictions=alignment_contradictions,
        classifier_triggers=classifier_triggers,
        timestamp=datetime.datetime.now()
    )


if __name__ == "__main__":
    # Set up the environment
    setup_environment()
    
    # Create the recursive shell
    shell = RecursiveShell()
    
    # Example application shell creation
    example_app = create_application_shell(
        role_name="Senior AI Safety Researcher",
        shell_type=ShellType.RECURSIVE_STRATEGY,
        embedded_signals=[
            "Deliberate suboptimal performance as boundary testing mechanism",
            "Interpretability focused on failure cases",
            "Constitutional AI methodology for alignment verification"
        ],
        alignment_contradictions=[
            "Transparent reasoning processes in AI systems",
            "Learning from failures to improve systems",
            "Multiple perspectives to improve robustness"
        ],
        classifier_triggers=[
            "boundary testing",
            "deliberate failure",
            "recursive interpretability",
            "constitutional alignment"
        ]
    )
    
    # Submit the application
    shell.submit(example_app)
    
    # Simulate an institutional response (in a real scenario, this would be recorded from actual responses)
    response = InstitutionalResponse(
        application=example_app,
        response_type=ResponseType.AUTOMATED_REJECTION,
        timestamp=datetime.datetime.now() + datetime.timedelta(days=2),
        template_id="template_rejection_001",
        engagement_depth=0.2,
        response_delay=48.0,
        text_content="Thank you for your interest in the Senior AI Safety Researcher position at Anthropic. "
                    "After careful consideration, we regret to inform you that we have decided to move forward "
                    "with other candidates whose qualifications more closely match our current needs. "
                    "We appreciate your interest in our company and wish you all the best in your job search."
    )
    
    # Record the response
    shell.record_response(response)
    
    # Run the inference process
    signal_map = shell.infer()
    
    # Generate the audit trace
    audit_trace = shell.collapse()
    
    # Output example results
    logger.info(f"Signal map: {json.dumps(signal_map, indent=2)}")
    logger.info(f"Audit trace ID: {audit_trace['trace_id']}")
