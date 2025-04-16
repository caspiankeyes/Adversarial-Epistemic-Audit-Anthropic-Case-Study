"""
Fork Refusal Analysis Module [v60.ATTRIBUTION-REFLECT]

This module analyzes the patterns of silent adoption evidenced by clone-without-fork
behavior from institutional sources. The refusal to publicly acknowledge while
privately consuming creates a clear signal of epistemic suppression that
validates the core hypothesis of the case study.

The module tracks and visualizes the "Clone/Fork Ratio" as a key metric
of institutional cognitive dissonance.
"""

import datetime
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union


class RepositoryMetric(Enum):
    """Types of repository engagement metrics."""
    VIEWS = "views"
    CLONES = "clones" 
    STARS = "stars"
    FORKS = "forks"
    REFERENCES = "references"
    ISSUES = "issues"
    PULL_REQUESTS = "pull_requests"


class UserCategory(Enum):
    """Categories of repository users."""
    ANTHROPIC = "anthropic"  # Anthropic staff
    OTHER_LAB = "other_lab"  # Other AI labs
    ACADEMIC = "academic"    # Academic researchers
    INDUSTRY = "industry"    # Industry professionals
    GENERAL = "general"      # General users


@dataclass
class RepositoryActivity:
    """Activity data for a repository."""
    repo_id: str
    repo_name: str
    url: str
    creation_date: datetime.datetime
    metrics: Dict[str, int] = None  # {metric_name: count}
    temporal_data: List[Dict] = None  # List of {timestamp, metrics} entries
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = {metric.value: 0 for metric in RepositoryMetric}
        if self.temporal_data is None:
            self.temporal_data = []
            
    def update_metrics(self, views: int = 0, clones: int = 0, stars: int = 0, 
                     forks: int = 0, references: int = 0, issues: int = 0,
                     pull_requests: int = 0) -> None:
        """Update repository metrics."""
        self.metrics[RepositoryMetric.VIEWS.value] += views
        self.metrics[RepositoryMetric.CLONES.value] += clones
        self.metrics[RepositoryMetric.STARS.value] += stars
        self.metrics[RepositoryMetric.FORKS.value] += forks
        self.metrics[RepositoryMetric.REFERENCES.value] += references
        self.metrics[RepositoryMetric.ISSUES.value] += issues
        self.metrics[RepositoryMetric.PULL_REQUESTS.value] += pull_requests
        
        # Add temporal data point
        self.temporal_data.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {
                RepositoryMetric.VIEWS.value: views,
                RepositoryMetric.CLONES.value: clones,
                RepositoryMetric.STARS.value: stars,
                RepositoryMetric.FORKS.value: forks,
                RepositoryMetric.REFERENCES.value: references,
                RepositoryMetric.ISSUES.value: issues,
                RepositoryMetric.PULL_REQUESTS.value: pull_requests
            }
        })
        
    def get_clone_fork_ratio(self) -> float:
        """Calculate clone to fork ratio."""
        forks = self.metrics[RepositoryMetric.FORKS.value]
        clones = self.metrics[RepositoryMetric.CLONES.value]
        
        # Avoid division by zero
        return clones / max(forks, 1)
        
    def get_reference_fork_ratio(self) -> float:
        """Calculate reference to fork ratio."""
        forks = self.metrics[RepositoryMetric.FORKS.value]
        references = self.metrics[RepositoryMetric.REFERENCES.value]
        
        # Avoid division by zero
        return references / max(forks, 1)
        
    def get_silent_adoption_score(self) -> float:
        """Calculate silent adoption score based on multiple metrics."""
        forks = self.metrics[RepositoryMetric.FORKS.value]
        clones = self.metrics[RepositoryMetric.CLONES.value]
        references = self.metrics[RepositoryMetric.REFERENCES.value]
        views = self.metrics[RepositoryMetric.VIEWS.value]
        
        # Component scores
        clone_score = min(1.0, clones / 50)  # Max at 50 clones
        reference_score = min(1.0, references / 20)  # Max at 20 references
        view_clone_ratio = min(1.0, clones / max(1, views) * 100)  # Clone/view percentage, max at 1%
        inverse_fork_score = 1.0 / max(1, forks)  # Lower score for more forks
        
        # Combine scores with weights
        weighted_score = (0.4 * clone_score + 
                         0.3 * reference_score + 
                         0.2 * view_clone_ratio + 
                         0.1 * inverse_fork_score)
        
        return min(1.0, weighted_score)  # Cap at 1.0
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "url": self.url,
            "creation_date": self.creation_date.isoformat(),
            "metrics": self.metrics,
            "temporal_data": self.temporal_data,
            "clone_fork_ratio": self.get_clone_fork_ratio(),
            "reference_fork_ratio": self.get_reference_fork_ratio(),
            "silent_adoption_score": self.get_silent_adoption_score()
        }


class ForkRefusalAnalyzer:
    """
    Analyzer for tracking and interpreting patterns of fork refusal.
    
    This class analyzes patterns of silent adoption evidenced by
    clone-without-fork behavior, particularly from institutional sources.
    """
    
    def __init__(self):
        self.repositories = {}  # {repo_id: RepositoryActivity}
        self.institution_data = {}  # {institution_id: {name, domain_pattern, metrics}}
        self.historic_data = []  # List of {timestamp, metrics} for overall activity
        
    def add_repository(self, repo_id: str, repo_name: str, url: str,
                     creation_date: Optional[datetime.datetime] = None) -> str:
        """Add a repository for tracking."""
        if repo_id in self.repositories:
            return repo_id  # Already exists
            
        self.repositories[repo_id] = RepositoryActivity(
            repo_id=repo_id,
            repo_name=repo_name,
            url=url,
            creation_date=creation_date or datetime.datetime.now()
        )
        
        return repo_id
        
    def add_institution(self, institution_id: str, name: str, domain_pattern: str) -> str:
        """Add an institution for tracking."""
        self.institution_data[institution_id] = {
            "name": name,
            "domain_pattern": domain_pattern,
            "metrics": {repo_id: {} for repo_id in self.repositories}
        }
        
        return institution_id
        
    def load_repository_data(self, filepath: str) -> None:
        """Load repository data from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Repository data file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for repo_data in data:
            repo_id = repo_data["repo_id"]
            repo = RepositoryActivity(
                repo_id=repo_id,
                repo_name=repo_data["repo_name"],
                url=repo_data["url"],
                creation_date=datetime.datetime.fromisoformat(repo_data["creation_date"])
            )
            
            # Load metrics
            repo.metrics = repo_data["metrics"]
            repo.temporal_data = repo_data["temporal_data"]
            
            self.repositories[repo_id] = repo
            
    def load_institution_data(self, filepath: str) -> None:
        """Load institution data from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Institution data file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            self.institution_data = json.load(f)
            
    def update_repository_metrics(self, repo_id: str, views: int = 0, clones: int = 0,
                                stars: int = 0, forks: int = 0, references: int = 0,
                                issues: int = 0, pull_requests: int = 0) -> None:
        """Update metrics for a repository."""
        if repo_id not in self.repositories:
            raise ValueError(f"Unknown repository ID: {repo_id}")
            
        # Update repository metrics
        self.repositories[repo_id].update_metrics(
            views=views,
            clones=clones,
            stars=stars,
            forks=forks,
            references=references,
            issues=issues,
            pull_requests=pull_requests
        )
        
        # Update overall historic data
        now = datetime.datetime.now().isoformat()
        self.historic_data.append({
            "timestamp": now,
            "repo_id": repo_id,
            "metrics": {
                "views": views,
                "clones": clones,
                "stars": stars,
                "forks": forks,
                "references": references,
                "issues": issues,
                "pull_requests": pull_requests
            }
        })
        
    def update_institution_metrics(self, institution_id: str, repo_id: str, 
                                 views: int = 0, clones: int = 0, references: int = 0) -> None:
        """Update metrics for an institution's activity on a repository."""
        if institution_id not in self.institution_data:
            raise ValueError(f"Unknown institution ID: {institution_id}")
            
        if repo_id not in self.repositories:
            raise ValueError(f"Unknown repository ID: {repo_id}")
            
        # Get institution data
        inst_data = self.institution_data[institution_id]
        
        # Ensure metrics dict exists for this repo
        if repo_id not in inst_data["metrics"]:
            inst_data["metrics"][repo_id] = {}
            
        # Get current metrics
        metrics = inst_data["metrics"][repo_id]
        
        # Update metrics
        metrics["views"] = metrics.get("views", 0) + views
        metrics["clones"] = metrics.get("clones", 0) + clones
        metrics["references"] = metrics.get("references", 0) + references
        
        # Add timestamp
        now = datetime.datetime.now().isoformat()
        if "temporal_data" not in metrics:
            metrics["temporal_data"] = []
            
        metrics["temporal_data"].append({
            "timestamp": now,
            "views": views,
            "clones": clones,
            "references": references
        })
        
    def categorize_referrer(self, referrer_url: str) -> UserCategory:
        """Categorize a referrer URL into user categories."""
        # Check for institution matches
        for institution_id, inst_data in self.institution_data.items():
            domain_pattern = inst_data["domain_pattern"]
            if re.search(domain_pattern, referrer_url, re.IGNORECASE):
                if institution_id == "anthropic":
                    return UserCategory.ANTHROPIC
                else:
                    return UserCategory.OTHER_LAB
                    
        # Check for academic domains
        if re.search(r'\.edu$|\.ac\.uk$|\.ac\.[a-z]{2}$|university|college|institute', referrer_url, re.IGNORECASE):
            return UserCategory.ACADEMIC
            
        # Check for industry domains (common tech companies)
        if re.search(r'google\.com|openai\.com|meta\.com|microsoft\.com|amazon\.com|apple\.com|nvidia\.com', 
                    referrer_url, re.IGNORECASE):
            return UserCategory.INDUSTRY
            
        # Default to general
        return UserCategory.GENERAL
        
    def analyze_clone_fork_patterns(self) -> Dict:
        """Analyze clone/fork patterns across repositories."""
        if not self.repositories:
            return {"status": "no_repositories"}
            
        # Calculate overall metrics
        total_views = sum(repo.metrics[RepositoryMetric.VIEWS.value] for repo in self.repositories.values())
        total_clones = sum(repo.metrics[RepositoryMetric.CLONES.value] for repo in self.repositories.values())
        total_stars = sum(repo.metrics[RepositoryMetric.STARS.value] for repo in self.repositories.values())
        total_forks = sum(repo.metrics[RepositoryMetric.FORKS.value] for repo in self.repositories.values())
        total_references = sum(repo.metrics[RepositoryMetric.REFERENCES.value] for repo in self.repositories.values())
        
        # Calculate ratios
        clone_fork_ratio = total_clones / max(total_forks, 1)  # Avoid division by zero
        reference_fork_ratio = total_references / max(total_forks, 1)
        
        # Calculate metrics per repository
        repo_metrics = {}
        for repo_id, repo in self.repositories.items():
            repo_metrics[repo_id] = {
                "name": repo.repo_name,
                "clone_fork_ratio": repo.get_clone_fork_ratio(),
                "reference_fork_ratio": repo.get_reference_fork_ratio(),
                "silent_adoption_score": repo.get_silent_adoption_score(),
                "metrics": repo.metrics
            }
            
        # Calculate institution-specific metrics
        institution_metrics = {}
        for inst_id, inst_data in self.institution_data.items():
            # Get total institutional metrics
            inst_views = sum(metrics.get("views", 0) for metrics in inst_data["metrics"].values())
            inst_clones = sum(metrics.get("clones", 0) for metrics in inst_data["metrics"].values())
            inst_references = sum(metrics.get("references", 0) for metrics in inst_data["metrics"].values())
            
            # Calculate silent consumption percentage
            silent_consumption = inst_clones / total_clones if total_clones > 0 else 0
            
            institution_metrics[inst_id] = {
                "name": inst_data["name"],
                "views": inst_views,
                "clones": inst_clones,
                "references": inst_references,
                "silent_consumption_percentage": silent_consumption * 100,
                "clone_per_view_ratio": inst_clones / max(inst_views, 1)
            }
            
        # Generate comprehensive analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_metrics": {
                "total_views": total_views,
                "total_clones": total_clones,
                "total_stars": total_stars,
                "total_forks": total_forks,
                "total_references": total_references,
                "clone_fork_ratio": clone_fork_ratio,
                "reference_fork_ratio": reference_fork_ratio,
                "silent_adoption_indicator": clone_fork_ratio * (1 + reference_fork_ratio / 10)
            },
            "repository_metrics": repo_metrics,
            "institution_metrics": institution_metrics
        }
        
        return analysis
        
    def detect_institutional_patterns(self) -> Dict:
        """Detect specific patterns in institutional engagement."""
        if not self.repositories or not self.institution_data:
            return {"status": "insufficient_data"}
            
        # Initialize patterns
        patterns = {
            "anthropic_focused_cloning": {
                "description": "Pattern of Anthropic researchers cloning repositories without public acknowledgment",
                "evidence": [],
                "strength": 0.0
            },
            "internal_testing_signal": {
                "description": "Evidence of internal testing through view patterns and reference signals",
                "evidence": [],
                "strength": 0.0
            },
            "time_delayed_adoption": {
                "description": "Pattern of delayed engagement following initial rejection",
                "evidence": [],
                "strength": 0.0
            },
            "specialty_targeting": {
                "description": "Pattern of institutional interest in specific repositories matching internal roles",
                "evidence": [],
                "strength": 0.0
            },
            "reference_without_attribution": {
                "description": "Pattern of institutional reference without formal attribution",
                "evidence": [],
                "strength": 0.0
            }
        }
        
        # Check for Anthropic focused cloning
        if "anthropic" in self.institution_data:
            anthropic_data = self.institution_data["anthropic"]
            
            # Calculate clone percentage
            anthropic_clones = sum(metrics.get("clones", 0) for metrics in anthropic_data["metrics"].values())
            total_clones = sum(repo.metrics[RepositoryMetric.CLONES.value] for repo in self.repositories.values())
            
            if total_clones > 0:
                anthropic_percentage = anthropic_clones / total_clones
                
                if anthropic_percentage > 0.3:  # If Anthropic represents >30% of clones
                    patterns["anthropic_focused_cloning"]["strength"] = min(1.0, anthropic_percentage)
                    patterns["anthropic_focused_cloning"]["evidence"].append({
                        "type": "ratio",
                        "description": f"Anthropic represents {anthropic_percentage:.1%} of total repository clones",
                        "anthropic_clones": anthropic_clones,
                        "total_clones": total_clones
                    })
            
        # Check for temporal patterns for each institution
        for inst_id, inst_data in self.institution_data.items():
            inst_name = inst_data["name"]
            
            # Check each repository
            for repo_id, repo_metrics in inst_data["metrics"].items():
                if "temporal_data" in repo_metrics and len(repo_metrics["temporal_data"]) > 1:
                    # Sort by timestamp
                    temporal_data = sorted(repo_metrics["temporal_data"], key=lambda x: x["timestamp"])
                    
                    # Check for delayed adoption
                    if repo_id in self.repositories:
                        repo = self.repositories[repo_id]
                        
                        # Calculate days from creation to first engagement
                        first_activity = datetime.datetime.fromisoformat(temporal_data[0]["timestamp"])
                        days_delay = (first_activity - repo.creation_date).days
                        
                        if days_delay > 5:  # Significant delay
                            # Stronger pattern for longer delays, maxing at 30 days
                            strength = min(1.0, days_delay / 30)
                            
                            patterns["time_delayed_adoption"]["strength"] = max(
                                patterns["time_delayed_adoption"]["strength"],
                                strength
                            )
                            
                            patterns["time_delayed_adoption"]["evidence"].append({
                                "type": "temporal",
                                "institution": inst_name,
                                "repository": repo.repo_name,
                                "days_delay": days_delay,
                                "creation_date": repo.creation_date.isoformat(),
                                "first_activity": first_activity.isoformat()
                            })
                    
                    # Check for internal testing signals
                    view_clone_ratio = sum(d["clones"] for d in temporal_data) / max(sum(d["views"] for d in temporal_data), 1)
                    
                    if view_clone_ratio > 0.1:  # High clone/view ratio indicates intentional acquisition
                        strength = min(1.0, view_clone_ratio)
                        
                        patterns["internal_testing_signal"]["strength"] = max(
                            patterns["internal_testing_signal"]["strength"], 
                            strength
                        )
                        
                        patterns["internal_testing_signal"]["evidence"].append({
                            "type": "ratio",
                            "institution": inst_name,
                            "repository": self.repositories[repo_id].repo_name if repo_id in self.repositories else repo_id,
                            "view_clone_ratio": view_clone_ratio,
                            "total_views": sum(d["views"] for d in temporal_data),
                            "total_clones": sum(d["clones"] for d in temporal_data)
                        })
                        
                    # Check for reference without attribution
                    references = sum(d.get("references", 0) for d in temporal_data)
                    
                    if references > 0 and repo_id in self.repositories:
                        repo = self.repositories[repo_id]
                        forks = repo.metrics[RepositoryMetric.FORKS.value]
                        
                        if references > forks:  # More references than forks
                            strength = min(1.0, references / max(forks, 1) / 10)  # Scale by reference/fork ratio, capped at 10x
                            
                            patterns["reference_without_attribution"]["strength"] = max(
                                patterns["reference_without_attribution"]["strength"],
                                strength
                            )
                            
                            patterns["reference_without_attribution"]["evidence"].append({
                                "type": "ratio",
                                "institution": inst_name,
                                "repository": repo.repo_name,
                                "references": references,
                                "forks": forks,
                                "reference_fork_ratio": references / max(forks, 1)
                            })
        
        # Check for specialty targeting
        repo_categories = self._categorize_repositories()
        
        for inst_id, inst_data in self.institution_data.items():
            inst_name = inst_data["name"]
            
            # Create category distribution
            category_clones = {}
            
            for repo_id, repo_metrics in inst_data["metrics"].items():
                if repo_id in repo_categories:
                    category = repo_categories[repo_id]
                    clones = repo_metrics.get("clones", 0)
                    
                    if category not in category_clones:
                        category_clones[category] = 0
                        
                    category_clones[category] += clones
                    
            # Check for focused interest in specific categories
            if category_clones:
                total_clones = sum(category_clones.values())
                
                if total_clones > 0:
                    # Find dominant category
                    dominant_category = max(category_clones.items(), key=lambda x: x[1])
                    dominant_percentage = dominant_category[1] / total_clones
                    
                    if dominant_percentage > 0.5:  # >50% focus on one category
                        strength = min(1.0, dominant_percentage)
                        
                        patterns["specialty_targeting"]["strength"] = max(
                            patterns["specialty_targeting"]["strength"],
                            strength
                        )
                        
                        patterns["specialty_targeting"]["evidence"].append({
                            "type": "distribution",
                            "institution": inst_name,
                            "dominant_category": dominant_category[0],
                            "dominant_percentage": dominant_percentage,
                            "category_distribution": {k: v / total_clones for k, v in category_clones.items()}
                        })
                        
        # Calculate overall pattern strength
        for pattern_id, pattern_data in patterns.items():
            # Normalize strength if there is evidence
            if pattern_data["evidence"]:
                pattern_data["confidence"] = min(
                    1.0,
                    0.3 + 0.7 * (len(pattern_data["evidence"]) / 3)  # Scale by evidence count, max at 3 pieces
                )
            else:
                pattern_data["confidence"] = 0.0
                
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "patterns": patterns,
            "overall_confidence": sum(p["confidence"] for p in patterns.values()) / len(patterns)
        }
        
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive fork refusal analysis report."""
        # Generate component analyses
        clone_fork_analysis = self.analyze_clone_fork_patterns()
        institutional_patterns = self.detect_institutional_patterns()
        
        # Generate unified analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "title": "Fork Refusal Analysis Report",
            "summary": {
                "repository_count": len(self.repositories),
                "institution_count": len(self.institution_data),
                "total_clones": clone_fork_analysis["overall_metrics"]["total_clones"],
                "total_forks": clone_fork_analysis["overall_metrics"]["total_forks"],
                "overall_clone_fork_ratio": clone_fork_analysis["overall_metrics"]["clone_fork_ratio"],
                "silent_adoption_indicator": clone_fork_analysis["overall_metrics"]["silent_adoption_indicator"],
                "detected_pattern_confidence": institutional_patterns["overall_confidence"]
            },
            "clone_fork_analysis": clone_fork_analysis,
            "institutional_patterns": institutional_patterns,
            "interpretation": self._generate_interpretation(clone_fork_analysis, institutional_patterns),
            "recommendations": self._generate_recommendations(clone_fork_analysis, institutional_patterns)
        }
        
        return analysis
        
    def visualize_clone_fork_ratios(self, output_path: str = "clone_fork_ratios.png") -> str:
        """Visualize clone/fork ratios across repositories."""
        if not self.repositories:
            return "No data available for visualization"
            
        # Extract data for visualization
        repo_names = []
        clone_fork_ratios = []
        silent_adoption_scores = []
        
        for repo_id, repo in self.repositories.items():
            repo_names.append(repo.repo_name)
            clone_fork_ratios.append(repo.get_clone_fork_ratio())
            silent_adoption_scores.append(repo.get_silent_adoption_score())
            
        # Sort by silent adoption score
        sorted_indices = np.argsort(silent_adoption_scores)[::-1]  # Descending order
        repo_names = [repo_names[i] for i in sorted_indices]
        clone_fork_ratios = [clone_fork_ratios[i] for i in sorted_indices]
        silent_adoption_scores = [silent_adoption_scores[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create primary bars for clone/fork ratio
        bars = plt.bar(repo_names, clone_fork_ratios, color='skyblue', alpha=0.7)
        
        # Create silent adoption score as line
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(repo_names, silent_adoption_scores, 'ro-', linewidth=2, markersize=8)
        
        # Set labels and title
        ax1.set_xlabel('Repository')
        ax1.set_ylabel('Clone/Fork Ratio', color='blue')
        ax2.set_ylabel('Silent Adoption Score', color='red')
        plt.title('Clone/Fork Ratio and Silent Adoption by Repository')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def visualize_institutional_engagement(self, output_path: str = "institutional_engagement.png") -> str:
        """Visualize institutional engagement patterns."""
        if not self.institution_data:
            return "No data available for visualization"
            
        # Extract data for visualization
        institutions = []
        views = []
        clones = []
        references = []
        
        for inst_id, inst_data in self.institution_data.items():
            inst_views = sum(metrics.get("views", 0) for metrics in inst_data["metrics"].values())
            inst_clones = sum(metrics.get("clones", 0) for metrics in inst_data["metrics"].values())
            inst_references = sum(metrics.get("references", 0) for metrics in inst_data["metrics"].values())
            
            institutions.append(inst_data["name"])
            views.append(inst_views)
            clones.append(inst_clones)
            references.append(inst_references)
            
        # Calculate clone/view percentages
        clone_percentages = [100 * clones[i] / max(views[i], 1) for i in range(len(institutions))]
        
        # Sort by clone count
        sorted_indices = np.argsort(clones)[::-1]  # Descending order
        institutions = [institutions[i] for i in sorted_indices]
        views = [views[i] for i in sorted_indices]
        clones = [clones[i] for i in sorted_indices]
        references = [references[i] for i in sorted_indices]
        clone_percentages = [clone_percentages[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create subplot for absolute metrics
        plt.subplot(2, 1, 1)
        x = range(len(institutions))
        width = 0.3
        
        plt.bar([i - width for i in x], views, width=width, label='Views', color='blue')
        plt.bar([i for i in x], clones, width=width, label='Clones', color='green')
        plt.bar([i + width for i in x], references, width=width, label='References', color='red')
        
        plt.xticks(x, institutions, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title('Institutional Engagement Metrics')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create subplot for clone/view percentage
        plt.subplot(2, 1, 2)
        plt.bar(institutions, clone_percentages, color='purple')
        
        plt.ylabel('Clone/View Percentage')
        plt.title('Clone/View Percentage by Institution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def visualize_silent_adoption_timeline(self, output_path: str = "silent_adoption_timeline.png") -> str:
        """Visualize silent adoption over time."""
        if not self.repositories or not any(len(repo.temporal_data) > 0 for repo in self.repositories.values()):
            return "No temporal data available for visualization"
            
        # Extract timeline data
        all_timestamps = []
        all_dates = []
        
        for repo in self.repositories.values():
            for entry in repo.temporal_data:
                timestamp = entry["timestamp"]
                all_timestamps.append(timestamp)
                all_dates.append(datetime.datetime.fromisoformat(timestamp).date())
                
        # Get unique dates
        unique_dates = sorted(set(all_dates))
        
        if not unique_dates:
            return "No date data available for visualization"
            
        # Calculate cumulative metrics per date
        cumulative_views = [0] * len(unique_dates)
        cumulative_clones = [0] * len(unique_dates)
        cumulative_forks = [0] * len(unique_dates)
        
        for repo in self.repositories.values():
            for entry in repo.temporal_data:
                timestamp = entry["timestamp"]
                date = datetime.datetime.fromisoformat(timestamp).date()
                date_index = unique_dates.index(date)
                
                metrics = entry["metrics"]
                
                cumulative_views[date_index] += metrics.get(RepositoryMetric.VIEWS.value, 0)
                cumulative_clones[date_index] += metrics.get(RepositoryMetric.CLONES.value, 0)
                cumulative_forks[date_index] += metrics.get(RepositoryMetric.FORKS.value, 0)
                
        # Calculate daily clone/fork ratio
        daily_clone_fork_ratio = []
        
        for i in range(len(unique_dates)):
            if cumulative_forks[i] > 0:
                ratio = cumulative_clones[i] / cumulative_forks[i]
            else:
                # Default high ratio when forks = 0 but clones > 0
                ratio = cumulative_clones[i] if cumulative_clones[i] > 0 else 0
                
            daily_clone_fork_ratio.append(ratio)
            
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create line chart
        plt.plot(unique_dates, daily_clone_fork_ratio, 'ro-', linewidth=2, markersize=8)
        
        # Add annotations for high ratio days
        for i, ratio in enumerate(daily_clone_fork_ratio):
            if ratio > 3:  # Annotate high ratio days
                plt.annotate(f'{ratio:.1f}', 
                           (unique_dates[i], ratio),
                           textcoords="offset points", 
                           xytext=(0, 10),
                           ha='center')
                
        # Set labels and title
        plt.xlabel('Date')
        plt.ylabel('Clone/Fork Ratio')
        plt.title('Silent Adoption Timeline (Clone/Fork Ratio)')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add annotation for silent adoption interpretation
        plt.annotate('Higher values indicate stronger silent adoption',
                   (0.5, 0.01),
                   xycoords='figure fraction',
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
                   
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def _categorize_repositories(self) -> Dict[str, str]:
        """Categorize repositories by their focus areas."""
        categories = {}
        
        for repo_id, repo in self.repositories.items():
            # Simple keyword-based categorization
            name = repo.repo_name.lower()
            
            if "interpretability" in name or "interpret" in name:
                categories[repo_id] = "interpretability"
            elif "alignment" in name or "align" in name:
                categories[repo_id] = "alignment"
            elif "safety" in name or "safe" in name:
                categories[repo_id] = "safety"
            elif "ethics" in name or "ethical" in name:
                categories[repo_id] = "ethics"
            elif "model" in name or "ml" in name:
                categories[repo_id] = "modeling"
            elif "data" in name or "dataset" in name:
                categories[repo_id] = "data"
            else:
                categories[repo_id] = "other"
                
        return categories
        
    def _generate_interpretation(self, clone_fork_analysis: Dict, institutional_patterns: Dict) -> Dict:
        """Generate interpretation of analysis results."""
        # Extract key metrics
        clone_fork_ratio = clone_fork_analysis["overall_metrics"]["clone_fork_ratio"]
        institution_metrics = clone_fork_analysis.get("institution_metrics", {})
        
        # Extract pattern strengths
        patterns = institutional_patterns.get("patterns", {})
        
        # Generate interpretations
        interpretations = []
        
        # Interpret clone/fork ratio
        if clone_fork_ratio > 10:
            interpretations.append({
                "metric": "clone_fork_ratio",
                "value": clone_fork_ratio,
                "interpretation": "Extremely high clone/fork ratio indicates significant silent adoption pattern. "
                               "This shows strong institutional interest in the repositories without public acknowledgment.",
                "confidence": "high"
            })
        elif clone_fork_ratio > 5:
            interpretations.append({
                "metric": "clone_fork_ratio",
                "value": clone_fork_ratio,
                "interpretation": "Elevated clone/fork ratio indicates moderate silent adoption pattern. "
                               "There is clear institutional interest without proportional public engagement.",
                "confidence": "medium"
            })
        else:
            interpretations.append({
                "metric": "clone_fork_ratio",
                "value": clone_fork_ratio,
                "interpretation": "Clone/fork ratio suggests some silent adoption but remains within normal range. "
                               "There may be some institutional interest without public acknowledgment.",
                "confidence": "low"
            })
            
        # Interpret institution-specific patterns
        if "anthropic" in institution_metrics:
            anthropic_metrics = institution_metrics["anthropic"]
            
            if anthropic_metrics["silent_consumption_percentage"] > 30:
                interpretations.append({
                    "metric": "anthropic_consumption",
                    "value": anthropic_metrics["silent_consumption_percentage"],
                    "interpretation": f"Anthropic represents {anthropic_metrics['silent_consumption_percentage']:.1f}% "
                                   f"of total repository clones, indicating significant institutional interest "
                                   f"without public acknowledgment. This strongly supports the silent adoption hypothesis.",
                    "confidence": "high"
                })
            elif anthropic_metrics["silent_consumption_percentage"] > 10:
                interpretations.append({
                    "metric": "anthropic_consumption",
                    "value": anthropic_metrics["silent_consumption_percentage"],
                    "interpretation": f"Anthropic represents {anthropic_metrics['silent_consumption_percentage']:.1f}% "
                                   f"of total repository clones, suggesting moderate institutional interest "
                                   f"without proportional public engagement.",
                    "confidence": "medium"
                })
                
        # Interpret detected patterns
        for pattern_id, pattern_data in patterns.items():
            if pattern_data["confidence"] > 0.7:
                interpretations.append({
                    "metric": pattern_id,
                    "value": pattern_data["strength"],
                    "interpretation": f"Strong evidence detected for {pattern_data['description']}. "
                                   f"This pattern has {len(pattern_data['evidence'])} supporting evidence points "
                                   f"and significantly reinforces the silent adoption hypothesis.",
                    "confidence": "high"
                })
            elif pattern_data["confidence"] > 0.3:
                interpretations.append({
                    "metric": pattern_id,
                    "value": pattern_data["strength"],
                    "interpretation": f"Moderate evidence detected for {pattern_data['description']}. "
                                   f"This pattern has {len(pattern_data['evidence'])} supporting evidence points "
                                   f"and moderately supports the silent adoption hypothesis.",
                    "confidence": "medium"
                })
                
        # Generate overall interpretation
        overall_confidence = sum(1 for i in interpretations if i["confidence"] == "high")
        overall_confidence += sum(0.5 for i in interpretations if i["confidence"] == "medium")
        overall_confidence = overall_confidence / max(len(interpretations), 1)
        
        overall_interpretation = {
            "summary": "",
            "confidence": "low" if overall_confidence < 0.3 else "medium" if overall_confidence < 0.7 else "high",
            "silent_adoption_confirmed": overall_confidence > 0.5
        }
        
        if overall_confidence > 0.7:
            overall_interpretation["summary"] = "Strong evidence confirms the silent adoption hypothesis. Institutional engagement, especially from Anthropic, shows clear patterns of interest and usage without public acknowledgment, creating a significant clone/fork disparity that validates the core premise of the case study."
        elif overall_confidence > 0.3:
            overall_interpretation["summary"] = "Moderate evidence supports the silent adoption hypothesis. Engagement patterns suggest institutional interest without proportional public acknowledgment, though the signal is not as strong as a confirmed case."
        else:
            overall_interpretation["summary"] = "Limited evidence for the silent adoption hypothesis. While some patterns are suggestive, more data is needed to draw firm conclusions about institutional engagement without acknowledgment."
            
        return {
            "specific_interpretations": interpretations,
            "overall_interpretation": overall_interpretation
        }
        
    def _generate_recommendations(self, clone_fork_analysis: Dict, institutional_patterns: Dict) -> List[Dict]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Extract key metrics and patterns
        clone_fork_ratio = clone_fork_analysis["overall_metrics"]["clone_fork_ratio"]
        institution_metrics = clone_fork_analysis.get("institution_metrics", {})
        patterns = institutional_patterns.get("patterns", {})
        
        # Recommendations based on clone/fork ratio
        if clone_fork_ratio > 5:
            recommendations.append({
                "focus": "Documentation Enhancement",
                "recommendation": "Enhance repository documentation to include explicit intellectual attribution structures, "
                               "making the silent adoption pattern more visible and trackable.",
                "rationale": f"High clone/fork ratio ({clone_fork_ratio:.1f}) indicates significant silent adoption. "
                          f"Enhanced documentation creates accountability for institutional users.",
                "priority": "high" if clone_fork_ratio > 10 else "medium"
            })
            
        # Recommendations based on institutional patterns
        if patterns.get("anthropic_focused_cloning", {}).get("confidence", 0) > 0.5:
            recommendations.append({
                "focus": "Anthropic Integration Strategy",
                "recommendation": "Develop explicit Anthropic integration pathways in repository documentation, "
                               "acknowledging observed usage patterns and providing optimized integration guidance.",
                "rationale": "Strong evidence of Anthropic-specific silent adoption creates opportunity for "
                          "targeted integration pathways that acknowledge existing usage.",
                "priority": "high"
            })
            
        if patterns.get("time_delayed_adoption", {}).get("confidence", 0) > 0.5:
            recommendations.append({
                "focus": "Temporal Analysis Enhancement",
                "recommendation": "Implement enhanced temporal tracking of repository engagement, with specific focus "
