import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ClauseType(Enum):
    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification" 
    TERMINATION = "termination"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    PAYMENT = "payment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    CONFIDENTIALITY = "confidentiality"

@dataclass
class RuleMatch:
    rule_id: str
    clause_type: ClauseType
    risk_level: RiskLevel
    confidence: float
    matched_text: str
    recommendation: str
    keywords_found: List[str]

class RulesetB:
    """
    Rule-based clause analysis system serving as fallback for RAG-Fusion.
    Provides comprehensive clause analysis using pattern matching and business rules.
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.risk_keywords = self._load_risk_keywords()
        self.compliance_patterns = self._load_compliance_patterns()
        self.contractual_patterns = self._load_contractual_patterns()
        
    def analyze_clauses(self, document: str, context: str = "") -> dict:
        """
        Main entry point for rule-based clause analysis.
        
        Args:
            document (str): Document text to analyze
            context (str): Additional context information
            
        Returns:
            dict: Analysis results with confidence, rules applied, and recommendations
        """
        try:
            # Clean and prepare document
            cleaned_doc = self._preprocess_document(document)
            
            # Apply all rule categories
            rule_matches = []
            rule_matches.extend(self._apply_risk_rules(cleaned_doc))
            rule_matches.extend(self._apply_compliance_rules(cleaned_doc))
            rule_matches.extend(self._apply_contractual_rules(cleaned_doc))
            rule_matches.extend(self._apply_ip_rules(cleaned_doc))
            rule_matches.extend(self._apply_privacy_rules(cleaned_doc))
            
            # Calculate overall analysis
            analysis_result = self._synthesize_analysis(rule_matches, cleaned_doc)
            
            return {
                "analysis": analysis_result,
                "method": "rule_based",
                "confidence": analysis_result["overall_confidence"],
                "rules_applied": [match.rule_id for match in rule_matches],
                "clause_matches": len(rule_matches),
                "risk_distribution": self._calculate_risk_distribution(rule_matches)
            }
            
        except Exception as e:
            return self._handle_analysis_error(str(e))

    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize all rule definitions."""
        return {
            "risk_rules": self._define_risk_rules(),
            "compliance_rules": self._define_compliance_rules(),
            "contractual_rules": self._define_contractual_rules(),
            "ip_rules": self._define_ip_rules(),
            "privacy_rules": self._define_privacy_rules()
        }

    def _define_risk_rules(self) -> List[Dict]:
        """Define risk-related clause analysis rules."""
        return [
            {
                "id": "RISK_001",
                "name": "Unlimited Liability",
                "patterns": [
                    r"unlimited\s+liabilit(y|ies)",
                    r"without\s+limit\s+.*liabilit(y|ies)",
                    r"no\s+cap\s+.*liabilit(y|ies)"
                ],
                "keywords": ["unlimited", "liability", "without limit", "no cap"],
                "risk_level": RiskLevel.CRITICAL,
                "clause_type": ClauseType.LIABILITY,
                "confidence": 0.9,
                "recommendation": "Negotiate liability caps to limit exposure"
            },
            {
                "id": "RISK_002", 
                "name": "Indemnification Scope",
                "patterns": [
                    r"indemnif(y|ication).*third.{0,20}part(y|ies)",
                    r"defend.*hold\s+harmless",
                    r"broad\s+indemnif(y|ication)"
                ],
                "keywords": ["indemnify", "indemnification", "hold harmless", "defend"],
                "risk_level": RiskLevel.HIGH,
                "clause_type": ClauseType.INDEMNIFICATION,
                "confidence": 0.85,
                "recommendation": "Review indemnification scope and consider mutual indemnity"
            },
            {
                "id": "RISK_003",
                "name": "Termination for Convenience",
                "patterns": [
                    r"terminat(e|ion).*convenience",
                    r"terminat(e|ion).*without\s+cause",
                    r"immediate\s+terminat(e|ion)"
                ],
                "keywords": ["termination", "convenience", "without cause", "immediate"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.TERMINATION,
                "confidence": 0.8,
                "recommendation": "Negotiate notice periods and termination fees"
            },
            {
                "id": "RISK_004",
                "name": "Penalty Clauses",
                "patterns": [
                    r"penalt(y|ies).*\$[\d,]+",
                    r"liquidated\s+damages.*\$[\d,]+",
                    r"fine.*\$[\d,]+"
                ],
                "keywords": ["penalty", "penalties", "liquidated damages", "fine"],
                "risk_level": RiskLevel.HIGH,
                "clause_type": ClauseType.PAYMENT,
                "confidence": 0.85,
                "recommendation": "Assess penalty amounts and negotiate caps"
            }
        ]

    def _define_compliance_rules(self) -> List[Dict]:
        """Define compliance-related rules."""
        return [
            {
                "id": "COMP_001",
                "name": "GDPR Compliance",
                "patterns": [
                    r"gdpr|general\s+data\s+protection\s+regulation",
                    r"data\s+subject\s+rights",
                    r"right\s+to\s+be\s+forgotten"
                ],
                "keywords": ["gdpr", "data protection", "data subject", "forgotten"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.PRIVACY,
                "confidence": 0.9,
                "recommendation": "Ensure GDPR compliance mechanisms are in place"
            },
            {
                "id": "COMP_002",
                "name": "SOX Compliance",
                "patterns": [
                    r"sarbanes.{0,10}oxley|sox\s+compliance",
                    r"section\s+404|section\s+302",
                    r"internal\s+controls.*financial\s+reporting"
                ],
                "keywords": ["sox", "sarbanes-oxley", "internal controls", "section 404"],
                "risk_level": RiskLevel.HIGH,
                "clause_type": ClauseType.COMPLIANCE,
                "confidence": 0.85,
                "recommendation": "Verify SOX compliance requirements and controls"
            },
            {
                "id": "COMP_003",
                "name": "Industry Standards",
                "patterns": [
                    r"iso\s+27001|iso\s+9001",
                    r"soc\s+[12]\s+type\s+[12]",
                    r"pci\s+dss|payment\s+card\s+industry"
                ],
                "keywords": ["iso 27001", "iso 9001", "soc", "pci dss"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.COMPLIANCE,
                "confidence": 0.8,
                "recommendation": "Verify compliance with specified industry standards"
            }
        ]

    def _define_contractual_rules(self) -> List[Dict]:
        """Define contractual obligation rules."""
        return [
            {
                "id": "CONT_001",
                "name": "Service Level Agreements",
                "patterns": [
                    r"sla|service\s+level\s+agreement",
                    r"uptime.*\d+%",
                    r"availability.*\d+%"
                ],
                "keywords": ["sla", "service level", "uptime", "availability"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.COMPLIANCE,
                "confidence": 0.8,
                "recommendation": "Review SLA commitments and penalty structures"
            },
            {
                "id": "CONT_002",
                "name": "Payment Terms",
                "patterns": [
                    r"net\s+\d+\s+days",
                    r"payment.*due.*\d+\s+days",
                    r"late\s+fee.*\d+%"
                ],
                "keywords": ["net", "payment due", "late fee", "days"],
                "risk_level": RiskLevel.LOW,
                "clause_type": ClauseType.PAYMENT,
                "confidence": 0.85,
                "recommendation": "Confirm payment terms align with cash flow requirements"
            },
            {
                "id": "CONT_003",
                "name": "Dispute Resolution",
                "patterns": [
                    r"arbitrat(e|ion)",
                    r"mediat(e|ion)",
                    r"governing\s+law.*\w+\s+(state|country)"
                ],
                "keywords": ["arbitration", "mediation", "governing law", "jurisdiction"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.DISPUTE_RESOLUTION,
                "confidence": 0.8,
                "recommendation": "Review dispute resolution mechanisms and jurisdiction"
            }
        ]

    def _define_ip_rules(self) -> List[Dict]:
        """Define intellectual property rules."""
        return [
            {
                "id": "IP_001",
                "name": "IP Ownership",
                "patterns": [
                    r"intellectual\s+propert(y|ies).*owner(ship)?",
                    r"work\s+for\s+hire",
                    r"derivative\s+works"
                ],
                "keywords": ["intellectual property", "ownership", "work for hire", "derivative"],
                "risk_level": RiskLevel.HIGH,
                "clause_type": ClauseType.INTELLECTUAL_PROPERTY,
                "confidence": 0.9,
                "recommendation": "Clarify IP ownership and usage rights"
            },
            {
                "id": "IP_002",
                "name": "License Restrictions",
                "patterns": [
                    r"license.*restrict(ed|ions?)",
                    r"non.{0,10}exclusive\s+license",
                    r"limited\s+license"
                ],
                "keywords": ["license", "restricted", "non-exclusive", "limited"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.INTELLECTUAL_PROPERTY,
                "confidence": 0.8,
                "recommendation": "Review license scope and restrictions"
            }
        ]

    def _define_privacy_rules(self) -> List[Dict]:
        """Define privacy and data protection rules."""
        return [
            {
                "id": "PRIV_001",
                "name": "Data Processing",
                "patterns": [
                    r"personal\s+data.*process(ing)?",
                    r"data\s+controller.*processor",
                    r"data\s+processing\s+agreement"
                ],
                "keywords": ["personal data", "processing", "controller", "processor"],
                "risk_level": RiskLevel.HIGH,
                "clause_type": ClauseType.PRIVACY,
                "confidence": 0.85,
                "recommendation": "Ensure proper data processing agreements are in place"
            },
            {
                "id": "PRIV_002",
                "name": "Data Retention",
                "patterns": [
                    r"data\s+retention.*\d+\s+(days?|months?|years?)",
                    r"delet(e|ion).*personal\s+data",
                    r"retention\s+period"
                ],
                "keywords": ["data retention", "deletion", "retention period"],
                "risk_level": RiskLevel.MEDIUM,
                "clause_type": ClauseType.PRIVACY,
                "confidence": 0.8,
                "recommendation": "Review data retention periods and deletion procedures"
            }
        ]

    def _apply_risk_rules(self, document: str) -> List[RuleMatch]:
        """Apply risk-related rules to document."""
        matches = []
        for rule in self.rules["risk_rules"]:
            rule_matches = self._apply_single_rule(document, rule)
            matches.extend(rule_matches)
        return matches

    def _apply_compliance_rules(self, document: str) -> List[RuleMatch]:
        """Apply compliance rules to document."""
        matches = []
        for rule in self.rules["compliance_rules"]:
            rule_matches = self._apply_single_rule(document, rule)
            matches.extend(rule_matches)
        return matches

    def _apply_contractual_rules(self, document: str) -> List[RuleMatch]:
        """Apply contractual rules to document."""
        matches = []
        for rule in self.rules["contractual_rules"]:
            rule_matches = self._apply_single_rule(document, rule)
            matches.extend(rule_matches)
        return matches

    def _apply_ip_rules(self, document: str) -> List[RuleMatch]:
        """Apply IP rules to document."""
        matches = []
        for rule in self.rules["ip_rules"]:
            rule_matches = self._apply_single_rule(document, rule)
            matches.extend(rule_matches)
        return matches

    def _apply_privacy_rules(self, document: str) -> List[RuleMatch]:
        """Apply privacy rules to document."""
        matches = []
        for rule in self.rules["privacy_rules"]:
            rule_matches = self._apply_single_rule(document, rule)
            matches.extend(rule_matches)
        return matches

    def _apply_single_rule(self, document: str, rule: Dict) -> List[RuleMatch]:
        """Apply a single rule to the document."""
        matches = []
        document_lower = document.lower()
        
        # Check pattern matches
        for pattern in rule["patterns"]:
            regex_matches = re.finditer(pattern, document_lower, re.IGNORECASE | re.MULTILINE)
            for match in regex_matches:
                # Extract surrounding context (±100 characters)
                start = max(0, match.start() - 100)
                end = min(len(document), match.end() + 100)
                context = document[start:end].strip()
                
                # Find keywords in the match
                found_keywords = [kw for kw in rule["keywords"] 
                                if kw.lower() in match.group().lower()]
                
                rule_match = RuleMatch(
                    rule_id=rule["id"],
                    clause_type=ClauseType(rule["clause_type"].value),
                    risk_level=rule["risk_level"],
                    confidence=rule["confidence"],
                    matched_text=context,
                    recommendation=rule["recommendation"],
                    keywords_found=found_keywords
                )
                matches.append(rule_match)
        
        return matches

    def _synthesize_analysis(self, matches: List[RuleMatch], document: str) -> Dict[str, Any]:
        """Synthesize overall analysis from rule matches."""
        if not matches:
            return {
                "overall_risk": "LOW",
                "overall_confidence": 0.3,
                "summary": "No significant clause issues detected using rule-based analysis",
                "recommendations": ["Consider manual review for completeness"],
                "clause_breakdown": {},
                "risk_score": 1
            }
        
        # Calculate risk distribution
        risk_counts = {level: 0 for level in RiskLevel}
        clause_counts = {ctype: 0 for ctype in ClauseType}
        
        for match in matches:
            risk_counts[match.risk_level] += 1
            clause_counts[match.clause_type] += 1
        
        # Calculate overall risk score (1-10 scale)
        risk_score = (
            risk_counts[RiskLevel.LOW] * 1 +
            risk_counts[RiskLevel.MEDIUM] * 3 +
            risk_counts[RiskLevel.HIGH] * 7 +
            risk_counts[RiskLevel.CRITICAL] * 10
        ) / max(len(matches), 1)
        
        # Determine overall risk level
        if risk_score >= 7:
            overall_risk = "CRITICAL"
        elif risk_score >= 5:
            overall_risk = "HIGH"
        elif risk_score >= 3:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        # Calculate confidence (average of all matches)
        overall_confidence = sum(match.confidence for match in matches) / len(matches)
        
        # Generate recommendations
        recommendations = list(set([match.recommendation for match in matches]))
        
        # Create clause breakdown
        clause_breakdown = {}
        for clause_type in ClauseType:
            type_matches = [m for m in matches if m.clause_type == clause_type]
            if type_matches:
                clause_breakdown[clause_type.value] = {
                    "count": len(type_matches),
                    "highest_risk": max(m.risk_level.value for m in type_matches),
                    "avg_confidence": sum(m.confidence for m in type_matches) / len(type_matches)
                }
        
        return {
            "overall_risk": overall_risk,
            "overall_confidence": min(overall_confidence, 0.85),  # Cap rule-based confidence
            "risk_score": round(risk_score, 2),
            "summary": f"Identified {len(matches)} clause issues across {len(clause_breakdown)} categories",
            "recommendations": recommendations,
            "clause_breakdown": clause_breakdown,
            "total_matches": len(matches)
        }

    def _calculate_risk_distribution(self, matches: List[RuleMatch]) -> Dict[str, int]:
        """Calculate distribution of risk levels."""
        distribution = {level.name: 0 for level in RiskLevel}
        for match in matches:
            distribution[match.risk_level.name] += 1
        return distribution

    def _preprocess_document(self, document: str) -> str:
        """Clean and prepare document for analysis."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', document)
        # Remove special characters that might interfere with regex
        cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\$\%\(\)\-]', ' ', cleaned)
        return cleaned.strip()

    def _load_risk_keywords(self) -> List[str]:
        """Load risk-related keywords."""
        return [
            "liability", "indemnification", "penalty", "damages", "breach",
            "default", "termination", "cancellation", "force majeure",
            "liquidated damages", "consequential damages", "punitive damages"
        ]

    def _load_compliance_patterns(self) -> List[str]:
        """Load compliance-related patterns."""
        return [
            "gdpr", "data protection", "privacy policy", "sox", "sarbanes-oxley",
            "iso 27001", "soc 2", "pci dss", "hipaa", "ferpa", "ccpa"
        ]

    def _load_contractual_patterns(self) -> List[str]:
        """Load contractual obligation patterns."""
        return [
            "service level", "sla", "uptime", "availability", "performance",
            "payment terms", "net 30", "late fee", "dispute resolution",
            "governing law", "jurisdiction", "arbitration", "mediation"
        ]

    def _handle_analysis_error(self, error: str) -> Dict[str, Any]:
        """Handle analysis errors gracefully."""
        return {
            "analysis": {
                "error": "Rule-based analysis failed",
                "error_details": error,
                "overall_risk": "UNKNOWN",
                "overall_confidence": 0.0,
                "summary": "Analysis could not be completed due to system error"
            },
            "method": "error_fallback",
            "confidence": 0.0,
            "rules_applied": [],
            "clause_matches": 0
        }

    def add_custom_rule(self, category: str, rule: Dict) -> bool:
        """
        Add a custom rule to the system.
        
        Args:
            category (str): Rule category ('risk_rules', 'compliance_rules', etc.)
            rule (Dict): Rule definition
            
        Returns:
            bool: Success status
        """
        try:
            if category in self.rules:
                self.rules[category].append(rule)
                return True
            return False
        except Exception:
            return False

    def get_rule_stats(self) -> Dict[str, int]:
        """Get statistics about loaded rules."""
        stats = {}
        for category, rules in self.rules.items():
            stats[category] = len(rules)
        stats["total_rules"] = sum(stats.values())
        return stats

# Utility functions for external use
def create_custom_rule(rule_id: str, name: str, patterns: List[str], 
                      keywords: List[str], risk_level: RiskLevel,
                      clause_type: ClauseType, confidence: float,
                      recommendation: str) -> Dict:
    """Helper function to create custom rules."""
    return {
        "id": rule_id,
        "name": name,
        "patterns": patterns,
        "keywords": keywords,
        "risk_level": risk_level,
        "clause_type": clause_type,
        "confidence": confidence,
        "recommendation": recommendation
    }
