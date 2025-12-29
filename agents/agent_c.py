import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

class RiskCategory(Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational" 
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    REPUTATIONAL = "reputational"
    STRATEGIC = "strategic"
    TECHNICAL = "technical"
    SECURITY = "security"

class RiskSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ConfidenceLevel(Enum):
    LOW = "low"        # < 0.4
    MEDIUM = "medium"  # 0.4 - 0.7
    HIGH = "high"      # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9

@dataclass
class RiskTag:
    category: RiskCategory
    severity: RiskSeverity
    confidence: float
    keywords: List[str]
    context: str
    mitigation_suggested: str

@dataclass
class RiskAnalysisResult:
    overall_risk_score: float
    confidence: float
    risk_tags: List[RiskTag]
    summary: str
    recommendations: List[str]
    fallback_used: bool
    analysis_method: str
    metadata: Dict[str, Any]

class RiskAnalyzerAgent:
    """
    Advanced Risk Analyzer using LLM with intelligent tagging and confidence-based fallback.
    """
    
    def __init__(self, model=None, confidence_threshold=0.6, fallback_threshold=0.4):
        self.model = model  # e.g., Gemini 2.5 Pro
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold
        self.risk_patterns = self._initialize_risk_patterns()
        self.tag_templates = self._initialize_tag_templates()
        self.logger = self._setup_logging()
        
    def analyze_risks(self, document: str, context: str = "", 
                     analysis_type: str = "comprehensive") -> RiskAnalysisResult:
        """
        Main entry point for risk analysis with confidence-based fallback.
        
        Args:
            document (str): Document to analyze
            context (str): Additional context information
            analysis_type (str): Type of analysis ('comprehensive', 'quick', 'focused')
            
        Returns:
            RiskAnalysisResult: Complete analysis with confidence scoring
        """
        try:
            self.logger.info(f"Starting risk analysis: {analysis_type}")
            
            # Step 1: Initial LLM-based risk analysis
            llm_result = self._perform_llm_analysis(document, context, analysis_type)
            
            # Step 2: Confidence assessment
            confidence_assessment = self._assess_confidence(llm_result, document)
            
            # Step 3: Tag generation and validation
            risk_tags = self._generate_risk_tags(document, llm_result, confidence_assessment)
            
            # Step 4: Decide on fallback based on confidence
            if confidence_assessment["overall_confidence"] >= self.confidence_threshold:
                # High confidence - use LLM results
                return self._finalize_analysis(
                    llm_result, risk_tags, confidence_assessment, 
                    fallback_used=False, method="llm_primary"
                )
            elif confidence_assessment["overall_confidence"] >= self.fallback_threshold:
                # Medium confidence - hybrid approach
                hybrid_result = self._perform_hybrid_analysis(
                    document, context, llm_result, risk_tags
                )
                return self._finalize_analysis(
                    hybrid_result, risk_tags, confidence_assessment,
                    fallback_used=True, method="hybrid"
                )
            else:
                # Low confidence - fallback to pattern-based analysis
                fallback_result = self._perform_fallback_analysis(document, context)
                return self._finalize_analysis(
                    fallback_result, risk_tags, confidence_assessment,
                    fallback_used=True, method="pattern_fallback"
                )
                
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {str(e)}")
            return self._handle_analysis_error(e, document)

    def _perform_llm_analysis(self, document: str, context: str, 
                            analysis_type: str) -> Dict[str, Any]:
        """
        Perform primary LLM-based risk analysis using Gemini.
        """
        prompt = self._build_analysis_prompt(document, context, analysis_type)
        
        try:
            # FIXED: Use generate_content for Gemini
            response = self.model.generate_content(prompt)
            parsed_result = self._parse_llm_response(response.text)
            
            # Enhance with additional analysis if comprehensive
            if analysis_type == "comprehensive":
                parsed_result = self._enhance_comprehensive_analysis(
                    parsed_result, document, context
                )
            
            return parsed_result
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {str(e)}")
            raise

    def _build_analysis_prompt(self, document: str, context: str, 
                             analysis_type: str) -> str:
        """
        Build sophisticated prompt for LLM risk analysis.
        """
        base_prompt = f"""
        You are an expert risk analyst. Analyze the following document for potential risks.
        
        Document to analyze:
        {document[:2000]}{"..." if len(document) > 2000 else ""}
        
        Additional context:
        {context}
        
        Analysis type: {analysis_type}
        """
        
        if analysis_type == "comprehensive":
            prompt = base_prompt + """
            
            Provide a comprehensive risk analysis including:
            
            1. RISK IDENTIFICATION:
               - Financial risks (cost overruns, payment delays, penalties)
               - Operational risks (service disruptions, performance failures)
               - Legal risks (contract breaches, compliance violations)
               - Reputational risks (brand damage, public relations issues)
               - Strategic risks (competitive disadvantage, market changes)
               - Technical risks (system failures, integration issues)
               - Security risks (data breaches, unauthorized access)
            
            2. RISK ASSESSMENT:
               - Severity level (1-4 scale: Low, Medium, High, Critical)
               - Probability of occurrence (Low, Medium, High)
               - Potential impact description
               - Time horizon (immediate, short-term, long-term)
            
            3. RISK CATEGORIZATION:
               - Primary risk category
               - Secondary categories if applicable
               - Risk interdependencies
            
            4. CONFIDENCE INDICATORS:
               - Analysis confidence (0.0-1.0)
               - Key evidence supporting assessment
               - Assumptions made
               - Areas of uncertainty
            
            5. MITIGATION RECOMMENDATIONS:
               - Immediate actions required
               - Long-term risk management strategies
               - Alternative approaches to consider
            
            Return your analysis in JSON format with the following structure:
            {
                "risks": [
                    {
                        "category": "category_name",
                        "severity": 1-4,
                        "probability": "low/medium/high",
                        "description": "detailed description",
                        "impact": "impact description",
                        "evidence": ["key evidence points"],
                        "mitigation": "recommended actions",
                        "confidence": 0.0-1.0
                    }
                ],
                "overall_assessment": {
                    "risk_score": 0.0-10.0,
                    "primary_concerns": ["list of main concerns"],
                    "confidence": 0.0-1.0,
                    "analysis_quality": "assessment of analysis quality"
                },
                "recommendations": ["prioritized recommendations"],
                "assumptions": ["key assumptions made"],
                "limitations": ["analysis limitations"]
            }
            """
            
        elif analysis_type == "quick":
            prompt = base_prompt + """
            
            Provide a quick risk assessment focusing on:
            1. Top 3 most significant risks
            2. Overall risk level (1-10 scale)
            3. Immediate actions needed
            4. Confidence in assessment (0.0-1.0)
            
            Return in JSON format:
            {
                "top_risks": [{"category": "", "severity": 1-4, "description": ""}],
                "overall_risk": 1-10,
                "immediate_actions": ["actions"],
                "confidence": 0.0-1.0
            }
            """
            
        else:  # focused analysis
            prompt = base_prompt + """
            
            Focus on specific risk areas mentioned in the context.
            Provide detailed analysis of these specific risks with:
            1. Risk validation and refinement
            2. Impact assessment
            3. Mitigation strategies
            4. Confidence assessment
            
            Return detailed JSON analysis.
            """
        
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return self._validate_llm_response(parsed)
            else:
                # Fallback to structured parsing
                return self._parse_unstructured_response(response_text)
                
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response, using text parsing")
            return self._parse_unstructured_response(response_text)

    def _validate_llm_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize LLM response format.
        """
        validated = {
            "risks": [],
            "overall_assessment": {},
            "recommendations": [],
            "confidence": 0.5,
            "raw_response": parsed
        }
        
        # Validate risks array
        if "risks" in parsed and isinstance(parsed["risks"], list):
            for risk in parsed["risks"]:
                if isinstance(risk, dict):
                    validated_risk = {
                        "category": risk.get("category", "operational"),
                        "severity": max(1, min(4, risk.get("severity", 2))),
                        "description": risk.get("description", ""),
                        "impact": risk.get("impact", ""),
                        "mitigation": risk.get("mitigation", ""),
                        "confidence": max(0.0, min(1.0, risk.get("confidence", 0.5)))
                    }
                    validated["risks"].append(validated_risk)
        
        # Handle quick analysis format
        if "top_risks" in parsed:
            for risk in parsed.get("top_risks", []):
                validated_risk = {
                    "category": risk.get("category", "operational"),
                    "severity": max(1, min(4, risk.get("severity", 2))),
                    "description": risk.get("description", ""),
                    "impact": "",
                    "mitigation": "",
                    "confidence": parsed.get("confidence", 0.5)
                }
                validated["risks"].append(validated_risk)
        
        # Validate overall assessment
        if "overall_assessment" in parsed:
            assessment = parsed["overall_assessment"]
            validated["overall_assessment"] = {
                "risk_score": max(0.0, min(10.0, assessment.get("risk_score", 5.0))),
                "confidence": max(0.0, min(1.0, assessment.get("confidence", 0.5))),
                "primary_concerns": assessment.get("primary_concerns", [])
            }
        elif "overall_risk" in parsed:
            # Handle quick analysis format
            validated["overall_assessment"] = {
                "risk_score": max(0.0, min(10.0, parsed.get("overall_risk", 5.0))),
                "confidence": max(0.0, min(1.0, parsed.get("confidence", 0.5))),
                "primary_concerns": []
            }
        
        # Extract recommendations
        if "recommendations" in parsed:
            validated["recommendations"] = parsed["recommendations"]
        elif "immediate_actions" in parsed:
            validated["recommendations"] = parsed["immediate_actions"]
        
        # Calculate overall confidence
        risk_confidences = [r.get("confidence", 0.5) for r in validated["risks"]]
        overall_conf = validated["overall_assessment"].get("confidence", 0.5)
        
        if risk_confidences:
            validated["confidence"] = (sum(risk_confidences) / len(risk_confidences) + overall_conf) / 2
        else:
            validated["confidence"] = overall_conf
            
        return validated

    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """
        Parse unstructured LLM response as fallback.
        """
        # Extract risk information using patterns
        risks = []
        
        # Look for risk patterns
        risk_sections = re.findall(r'(?:risk|concern|issue).*?(?=\n\n|\n[A-Z]|$)', 
                                 response, re.IGNORECASE | re.DOTALL)
        
        for section in risk_sections:
            risk = {
                "category": self._extract_category(section),
                "severity": self._extract_severity(section),
                "description": section.strip(),
                "confidence": 0.4  # Lower confidence for parsed responses
            }
            risks.append(risk)
        
        return {
            "risks": risks,
            "overall_assessment": {"risk_score": 5.0, "confidence": 0.4},
            "recommendations": self._extract_recommendations(response),
            "confidence": 0.4,
            "parsing_method": "unstructured"
        }

    def _extract_category(self, text: str) -> str:
        """Extract risk category from text."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["cost", "budget", "financial", "money"]):
            return "financial"
        elif any(word in text_lower for word in ["legal", "contract", "compliance", "regulation"]):
            return "legal"
        elif any(word in text_lower for word in ["security", "breach", "data", "cyber"]):
            return "security"
        elif any(word in text_lower for word in ["operational", "service", "process", "system"]):
            return "operational"
        else:
            return "operational"

    def _extract_severity(self, text: str) -> int:
        """Extract severity level from text."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["critical", "severe", "major"]):
            return 4
        elif any(word in text_lower for word in ["high", "significant", "important"]):
            return 3
        elif any(word in text_lower for word in ["medium", "moderate", "notable"]):
            return 2
        else:
            return 1

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text."""
        recommendations = []
        
        # Look for recommendation patterns
        rec_patterns = [
            r'recommend(?:ed|ation)?.*?(?=\n|$)',
            r'suggest(?:ed|ion)?.*?(?=\n|$)',
            r'should.*?(?=\n|$)',
            r'consider.*?(?=\n|$)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            recommendations.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _assess_confidence(self, llm_result: Dict[str, Any], 
                         document: str) -> Dict[str, Any]:
        """
        Assess confidence in LLM analysis results.
        """
        confidence_factors = {
            "response_completeness": 0.0,
            "risk_specificity": 0.0,
            "evidence_quality": 0.0,
            "consistency": 0.0,
            "document_coverage": 0.0
        }
        
        # Factor 1: Response completeness
        required_fields = ["risks", "overall_assessment", "recommendations"]
        present_fields = sum(1 for field in required_fields if field in llm_result)
        confidence_factors["response_completeness"] = present_fields / len(required_fields)
        
        # Factor 2: Risk specificity
        if "risks" in llm_result:
            specific_risks = sum(1 for risk in llm_result["risks"] 
                               if len(risk.get("description", "").split()) > 10)
            total_risks = len(llm_result["risks"])
            confidence_factors["risk_specificity"] = specific_risks / max(total_risks, 1)
        
        # Factor 3: Evidence quality
        evidence_score = 0.0
        if "risks" in llm_result:
            for risk in llm_result["risks"]:
                if "evidence" in risk and risk["evidence"]:
                    evidence_score += 1
            evidence_score = evidence_score / max(len(llm_result["risks"]), 1)
        confidence_factors["evidence_quality"] = evidence_score
        
        # Factor 4: Consistency check
        if "overall_assessment" in llm_result and "risks" in llm_result:
            individual_confidences = [r.get("confidence", 0.5) for r in llm_result["risks"]]
            overall_confidence = llm_result["overall_assessment"].get("confidence", 0.5)
            
            if individual_confidences:
                avg_individual = sum(individual_confidences) / len(individual_confidences)
                consistency = 1.0 - abs(avg_individual - overall_confidence)
                confidence_factors["consistency"] = max(0.0, consistency)
        
        # Factor 5: Document coverage
        doc_words = set(document.lower().split())
        analysis_text = json.dumps(llm_result).lower()
        analysis_words = set(analysis_text.split())
        
        if doc_words:
            coverage = len(doc_words & analysis_words) / len(doc_words)
            confidence_factors["document_coverage"] = min(coverage * 2, 1.0)  # Scale up coverage
        
        # Calculate weighted overall confidence
        weights = {
            "response_completeness": 0.25,
            "risk_specificity": 0.20,
            "evidence_quality": 0.20,
            "consistency": 0.15,
            "document_coverage": 0.20
        }
        
        overall_confidence = sum(
            confidence_factors[factor] * weights[factor] 
            for factor in confidence_factors
        )
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_factors": confidence_factors,
            "confidence_level": self._determine_confidence_level(overall_confidence),
            "assessment_timestamp": datetime.now().isoformat()
        }

    def _generate_risk_tags(self, document: str, llm_result: Dict[str, Any], 
                          confidence_assessment: Dict[str, Any]) -> List[RiskTag]:
        """
        Generate intelligent risk tags based on analysis.
        """
        tags = []
        
        if "risks" in llm_result:
            for risk in llm_result["risks"]:
                try:
                    # Map category string to enum
                    category = self._map_to_risk_category(risk.get("category", "operational"))
                    
                    # Create risk tag
                    tag = RiskTag(
                        category=category,
                        severity=RiskSeverity(risk.get("severity", 2)),
                        confidence=risk.get("confidence", 0.5),
                        keywords=self._extract_keywords(risk.get("description", "")),
                        context=risk.get("description", ""),
                        mitigation_suggested=risk.get("mitigation", "")
                    )
                    tags.append(tag)
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Failed to create risk tag: {e}")
                    continue
        
        # Add pattern-based tags if confidence is low
        if confidence_assessment["overall_confidence"] < 0.6:
            pattern_tags = self._generate_pattern_tags(document)
            tags.extend(pattern_tags)
        
        return tags

    def _generate_pattern_tags(self, document: str) -> List[RiskTag]:
        """Generate risk tags using pattern matching."""
        tags = []
        
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, document, re.IGNORECASE)
                for match in matches:
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(document), match.end() + 50)
                    context = document[context_start:context_end].strip()
                    
                    tag = RiskTag(
                        category=self._map_to_risk_category(category),
                        severity=RiskSeverity.MEDIUM,  # Default for pattern-based
                        confidence=0.6,
                        keywords=[match.group(0).lower()],
                        context=context,
                        mitigation_suggested="Review and assess this risk area"
                    )
                    tags.append(tag)
        
        return tags[:5]  # Limit pattern tags

    def _perform_hybrid_analysis(self, document: str, context: str,
                               llm_result: Dict[str, Any], 
                               risk_tags: List[RiskTag]) -> Dict[str, Any]:
        """
        Perform hybrid analysis combining LLM and pattern-based approaches.
        """
        # Start with LLM results
        hybrid_result = llm_result.copy()
        
        # Enhance with pattern-based validation
        pattern_risks = self._identify_pattern_risks(document)
        
        # Cross-validate LLM risks with patterns
        validated_risks = []
        for llm_risk in llm_result.get("risks", []):
            validation_score = self._validate_risk_with_patterns(llm_risk, pattern_risks)
            llm_risk["validation_score"] = validation_score
            llm_risk["confidence"] *= (0.5 + validation_score * 0.5)  # Adjust confidence
            validated_risks.append(llm_risk)
        
        # Add high-confidence pattern risks not found by LLM
        for pattern_risk in pattern_risks:
            if pattern_risk["confidence"] > 0.8:
                if not self._risk_already_identified(pattern_risk, validated_risks):
                    validated_risks.append(pattern_risk)
        
        hybrid_result["risks"] = validated_risks
        hybrid_result["analysis_method"] = "hybrid"
        hybrid_result["validation_applied"] = True
        
        return hybrid_result

    def _identify_pattern_risks(self, document: str) -> List[Dict[str, Any]]:
        """Identify risks using pattern matching."""
        pattern_risks = []
        
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, document, re.IGNORECASE)
                for match in matches:
                    risk = {
                        "category": category,
                        "severity": 2,
                        "description": f"Pattern-identified risk: {match.group(0)}",
                        "confidence": 0.7,
                        "pattern_matched": pattern
                    }
                    pattern_risks.append(risk)
        
        return pattern_risks

    def _validate_risk_with_patterns(self, llm_risk: Dict[str, Any], 
                                   pattern_risks: List[Dict[str, Any]]) -> float:
        """Validate LLM-identified risk with pattern matching."""
        llm_category = llm_risk.get("category", "").lower()
        llm_description = llm_risk.get("description", "").lower()
        
        validation_score = 0.0
        for pattern_risk in pattern_risks:
            if pattern_risk["category"] == llm_category:
                validation_score += 0.3
            
            # Check if pattern keywords appear in LLM description
            pattern_keywords = pattern_risk.get("pattern_matched", "").split("|")
            for keyword in pattern_keywords:
                if keyword.lower() in llm_description:
                    validation_score += 0.2
        
        return min(validation_score, 1.0)

    def _risk_already_identified(self, new_risk: Dict[str, Any], 
                               existing_risks: List[Dict[str, Any]]) -> bool:
        """Check if a risk is already identified."""
        new_category = new_risk.get("category", "")
        new_desc = new_risk.get("description", "").lower()
        
        for existing in existing_risks:
            if existing.get("category") == new_category:
                existing_desc = existing.get("description", "").lower()
                # Simple similarity check
                common_words = set(new_desc.split()) & set(existing_desc.split())
                if len(common_words) > 2:
                    return True
        
        return False

    def _perform_fallback_analysis(self, document: str, context: str) -> Dict[str, Any]:
        """
        Perform pattern-based fallback analysis when LLM confidence is low.
        """
        self.logger.info("Performing pattern-based fallback analysis")
        
        # Use comprehensive pattern matching
        pattern_risks = self._comprehensive_pattern_analysis(document)
        
        # Calculate overall risk score
        if pattern_risks:
            severity_scores = [risk["severity"] for risk in pattern_risks]
            overall_score = sum(severity_scores) / len(severity_scores) * 2.5  # Scale to 1-10
        else:
            overall_score = 2.0  # Low risk if no patterns found
        
        return {
            "risks": pattern_risks,
            "overall_assessment": {
                "risk_score": min(overall_score, 10.0),
                "confidence": 0.6,  # Moderate confidence in pattern matching
                "analysis_method": "pattern_based"
            },
            "recommendations": self._generate_pattern_recommendations(pattern_risks),
            "confidence": 0.6,
            "fallback_reason": "Low LLM confidence"
        }

    def _comprehensive_pattern_analysis(self, document: str) -> List[Dict[str, Any]]:
        """Perform comprehensive pattern-based risk analysis."""
        risks = []
        
        for category, patterns in self.risk_patterns.items():
            category_risks = 0
            for pattern in patterns:
                matches = list(re.finditer(pattern, document, re.IGNORECASE))
                if matches and category_risks < 2:  # Limit per category
                    for match in matches[:2]:  # Max 2 per pattern
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(document), match.end() + 100)
                        context = document[context_start:context_end].strip()
                        
                        risk = {
                            "category": category,
                            "severity": self._assess_pattern_severity(match.group(0), context),
                            "description": f"Identified {category} risk: {match.group(0)} in context: {context[:100]}...",
                            "mitigation": f"Review and address {category} concerns",
                            "confidence": 0.7
                        }
                        risks.append(risk)
                        category_risks += 1
        
        return risks

    def _assess_pattern_severity(self, matched_text: str, context: str) -> int:
        """Assess severity based on pattern match and context."""
        text_lower = matched_text.lower() + " " + context.lower()
        
        if any(word in text_lower for word in ["critical", "severe", "major", "significant"]):
            return 4
        elif any(word in text_lower for word in ["high", "important", "substantial"]):
            return 3
        elif any(word in text_lower for word in ["medium", "moderate", "potential"]):
            return 2
        else:
            return 1

    def _generate_pattern_recommendations(self, pattern_risks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on pattern-identified risks."""
        recommendations = []
        
        categories_found = set(risk["category"] for risk in pattern_risks)
        
        for category in categories_found:
            if category == "financial":
                recommendations.append("Review financial terms and establish clear payment schedules")
            elif category == "legal":
                recommendations.append("Conduct legal review and ensure compliance requirements are met")
            elif category == "operational":
                recommendations.append("Assess operational procedures and establish contingency plans")
            elif category == "security":
                recommendations.append("Implement security measures and data protection protocols")
        
        if not recommendations:
            recommendations.append("Conduct comprehensive risk assessment")
        
        return recommendations

    def _finalize_analysis(self, analysis_result: Dict[str, Any], 
                         risk_tags: List[RiskTag],
                         confidence_assessment: Dict[str, Any],
                         fallback_used: bool, method: str) -> RiskAnalysisResult:
        """
        Finalize and package the complete risk analysis.
        """
        return RiskAnalysisResult(
            overall_risk_score=analysis_result.get("overall_assessment", {}).get("risk_score", 5.0),
            confidence=confidence_assessment["overall_confidence"],
            risk_tags=risk_tags,
            summary=self._generate_analysis_summary(analysis_result, risk_tags),
            recommendations=analysis_result.get("recommendations", []),
            fallback_used=fallback_used,
            analysis_method=method,
            metadata={
                "total_risks_identified": len(analysis_result.get("risks", [])),
                "confidence_factors": confidence_assessment.get("confidence_factors", {}),
                "analysis_timestamp": datetime.now().isoformat(),
                "document_length": len(str(analysis_result)),
                "tag_distribution": self._calculate_tag_distribution(risk_tags)
            }
        )

    def _generate_analysis_summary(self, analysis_result: Dict[str, Any], 
                                 risk_tags: List[RiskTag]) -> str:
        """Generate summary of risk analysis."""
        risks = analysis_result.get("risks", [])
        if not risks:
            return "No significant risks identified in the document."
        
        risk_count = len(risks)
        high_severity = len([r for r in risks if r.get("severity", 1) >= 3])
        
        categories = list(set(r.get("category", "unknown") for r in risks))
        primary_category = categories[0] if categories else "general"
        
        summary = f"Identified {risk_count} risks with {high_severity} high-severity items. "
        summary += f"Primary risk area: {primary_category}. "
        
        if analysis_result.get("overall_assessment", {}).get("risk_score", 0) > 7:
            summary += "Overall risk level is HIGH - immediate attention recommended."
        elif analysis_result.get("overall_assessment", {}).get("risk_score", 0) > 4:
            summary += "Overall risk level is MODERATE - regular monitoring advised."
        else:
            summary += "Overall risk level is LOW - standard precautions sufficient."
        
        return summary

    def _calculate_tag_distribution(self, risk_tags: List[RiskTag]) -> Dict[str, int]:
        """Calculate distribution of risk tags by category."""
        distribution = {}
        for tag in risk_tags:
            category = tag.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def _enhance_comprehensive_analysis(self, parsed_result: Dict[str, Any], 
                                      document: str, context: str) -> Dict[str, Any]:
        """Enhance comprehensive analysis with additional insights."""
        # Add document-specific insights
        doc_length = len(document)
        word_count = len(document.split())
        
        # Enhance metadata
        if "metadata" not in parsed_result:
            parsed_result["metadata"] = {}
        
        parsed_result["metadata"].update({
            "document_stats": {
                "length": doc_length,
                "word_count": word_count,
                "complexity": "high" if word_count > 1000 else "medium" if word_count > 500 else "low"
            },
            "analysis_depth": "comprehensive",
            "enhancement_applied": True
        })
        
        return parsed_result

    # Utility and helper methods
    def _initialize_risk_patterns(self) -> Dict[str, List[str]]:
        """Initialize risk detection patterns."""
        return {
            "financial": [
                r"cost\s+overrun", r"budget\s+exceed", r"financial\s+loss",
                r"penalty.*\$[\d,]+", r"late\s+payment", r"cash\s+flow",
                r"price\s+increase", r"additional\s+cost"
            ],
            "legal": [
                r"breach\s+of\s+contract", r"litigation", r"lawsuit",
                r"non.{0,10}compliance", r"regulatory\s+violation",
                r"legal\s+action", r"dispute"
            ],
            "operational": [
                r"service\s+disruption", r"system\s+failure", r"downtime",
                r"performance\s+degradation", r"operational\s+risk",
                r"delay", r"interruption"
            ],
            "security": [
                r"data\s+breach", r"security\s+incident", r"unauthorized\s+access",
                r"cyber\s+attack", r"vulnerability", r"confidential\s+information"
            ],
            "compliance": [
                r"regulatory\s+requirement", r"compliance\s+issue",
                r"audit", r"standard.*requirement"
            ]
        }

    def _initialize_tag_templates(self) -> Dict[str, Dict]:
        """Initialize risk tag templates."""
        return {
            "high_severity": {
                "keywords": ["critical", "severe", "major", "significant"],
                "patterns": [r"critical.*risk", r"severe.*impact", r"major.*concern"]
            },
            "immediate_action": {
                "keywords": ["immediate", "urgent", "asap", "critical"],
                "patterns": [r"immediate.*action", r"urgent.*attention"]
            }
        }

    def _map_to_risk_category(self, category_str: str) -> RiskCategory:
        """Map string to RiskCategory enum."""
        category_mapping = {
            "financial": RiskCategory.FINANCIAL,
            "operational": RiskCategory.OPERATIONAL,
            "legal": RiskCategory.LEGAL,
            "compliance": RiskCategory.COMPLIANCE,
            "reputational": RiskCategory.REPUTATIONAL,
            "strategic": RiskCategory.STRATEGIC,
            "technical": RiskCategory.TECHNICAL,
            "security": RiskCategory.SECURITY
        }
        
        category_lower = category_str.lower()
        for key, value in category_mapping.items():
            if key in category_lower:
                return value
        
        return RiskCategory.OPERATIONAL  # Default

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level from numeric confidence."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w{4,}\b', text.lower())
        risk_keywords = [word for word in words if word in 
                        ["risk", "threat", "danger", "concern", "issue", "problem",
                         "liability", "exposure", "vulnerability", "impact", "failure",
                         "breach", "violation", "penalty", "loss", "damage"]]
        return list(set(risk_keywords))[:10]  # Limit to 10 keywords

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent."""
        logger = logging.getLogger(f"{__name__}.RiskAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _handle_analysis_error(self, error: Exception, document: str) -> RiskAnalysisResult:
        """Handle analysis errors gracefully."""
        self.logger.error(f"Analysis error: {str(error)}")
        
        return RiskAnalysisResult(
            overall_risk_score=5.0,  # Neutral risk score
            confidence=0.0,
            risk_tags=[],
            summary=f"Risk analysis failed due to error: {str(error)[:100]}",
            recommendations=["Manual review required due to analysis failure"],
            fallback_used=True,
            analysis_method="error_fallback",
            metadata={
                "error": str(error),
                "error_timestamp": datetime.now().isoformat(),
                "document_length": len(document)
            }
        )

    # Additional helper methods
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold dynamically."""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics and performance metrics."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "fallback_threshold": self.fallback_threshold,
            "supported_categories": [cat.value for cat in RiskCategory],
            "risk_patterns_loaded": len(self.risk_patterns),
            "agent_status": "active"
        }
