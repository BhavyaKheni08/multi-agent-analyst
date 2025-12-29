import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

class TipCategory(Enum):
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    TECHNICAL = "technical"
    PROCESS = "process"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    COMPLIANCE = "compliance"

class TipType(Enum):
    BEST_PRACTICE = "best_practice"
    OPTIMIZATION = "optimization"
    AUTOMATION = "automation"
    WORKFLOW = "workflow"
    TOOL_USAGE = "tool_usage"
    GUIDELINE = "guideline"

class TipPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ActionableTip:
    title: str
    category: TipCategory
    tip_type: TipType
    priority: TipPriority
    description: str
    implementation_steps: List[str]
    expected_outcome: str
    time_to_implement: str
    resources_needed: List[str]
    confidence: float

@dataclass
class TipExtractionResult:
    total_tips_extracted: int
    confidence: float
    summary: str
    recommendations: List[str]
    category_distribution: Dict[str, int]
    extraction_method: str
    tips: List[ActionableTip]
    metadata: Dict[str, Any]

class YouTubeTipExtractorAgent:
    """
    Advanced Tip Extractor that generates actionable insights and recommendations.
    """
    
    def __init__(self, model=None, confidence_threshold=0.6):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.tip_patterns = self._initialize_tip_patterns()
        self.logger = self._setup_logging()
        
    def extract_tips(self, document: str, context: str = "", 
                    tip_count_limit: int = 15) -> TipExtractionResult:
        """
        Main entry point for tip extraction.
        
        Args:
            document (str): Document to analyze
            context (str): Additional context information
            tip_count_limit (int): Maximum number of tips to extract
            
        Returns:
            TipExtractionResult: Complete tip extraction results
        """
        try:
            self.logger.info(f"Starting tip extraction, limit: {tip_count_limit}")
            
            # Step 1: LLM-based tip generation
            llm_tips = self._generate_llm_tips(document, context, tip_count_limit)
            
            # Step 2: Pattern-based tip enhancement
            pattern_tips = self._generate_pattern_tips(document)
            
            # Step 3: Combine and prioritize tips
            combined_tips = self._combine_and_prioritize_tips(llm_tips, pattern_tips, tip_count_limit)
            
            # Step 4: Calculate confidence and generate summary
            confidence = self._calculate_overall_confidence(combined_tips)
            summary = self._generate_summary(combined_tips)
            recommendations = self._generate_recommendations(combined_tips)
            
            # Step 5: Calculate category distribution
            category_distribution = self._calculate_category_distribution(combined_tips)
            
            return TipExtractionResult(
                total_tips_extracted=len(combined_tips),
                confidence=confidence,
                summary=summary,
                recommendations=recommendations,
                category_distribution=category_distribution,
                extraction_method="hybrid_llm_pattern",
                tips=combined_tips,
                metadata={
                    "extraction_timestamp": datetime.now().isoformat(),
                    "document_length": len(document),
                    "context_provided": bool(context),
                    "requested_limit": tip_count_limit,
                    "llm_tips_generated": len(llm_tips),
                    "pattern_tips_generated": len(pattern_tips)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tip extraction error: {str(e)}")
            return self._handle_extraction_error(e, document)

    def _generate_llm_tips(self, document: str, context: str, limit: int) -> List[ActionableTip]:
        """
        Generate tips using LLM analysis.
        """
        try:
            prompt = self._build_tip_extraction_prompt(document, context, limit)
            
            # FIXED: Use generate_content for Gemini
            response = self.model.generate_content(prompt)
            parsed_tips = self._parse_llm_tips_response(response.text)
            
            return self._convert_to_actionable_tips(parsed_tips)
            
        except Exception as e:
            self.logger.warning(f"LLM tip generation failed: {str(e)}")
            return []

    def _build_tip_extraction_prompt(self, document: str, context: str, limit: int) -> str:
        """
        Build comprehensive prompt for tip extraction.
        """
        prompt = f"""
        You are an expert consultant specializing in process optimization and best practices. 
        Analyze the following document and extract {limit} actionable tips for improvement.

        Document to analyze:
        {document[:2000]}{"..." if len(document) > 2000 else ""}

        Additional context:
        {context}

        For each tip, provide:
        1. Clear, actionable title
        2. Category (productivity, communication, technical, process, quality, efficiency, collaboration, compliance)
        3. Type (best_practice, optimization, automation, workflow, tool_usage, guideline)
        4. Priority level (1=low, 2=medium, 3=high, 4=critical)
        5. Detailed description
        6. Step-by-step implementation instructions
        7. Expected outcomes/benefits
        8. Time estimate to implement
        9. Resources needed
        10. Confidence level (0.0-1.0)

        Focus on:
        - Practical, implementable suggestions
        - Process improvements
        - Efficiency gains
        - Risk mitigation
        - Quality enhancements
        - Communication improvements
        - Automation opportunities

        Return your analysis in JSON format:
        {{
            "tips": [
                {{
                    "title": "Clear actionable title",
                    "category": "category_name",
                    "type": "tip_type",
                    "priority": 1-4,
                    "description": "Detailed description",
                    "implementation_steps": ["Step 1", "Step 2", "..."],
                    "expected_outcome": "What will be achieved",
                    "time_to_implement": "Time estimate",
                    "resources_needed": ["Resource 1", "Resource 2"],
                    "confidence": 0.0-1.0
                }}
            ],
            "extraction_confidence": 0.0-1.0,
            "analysis_notes": "Additional insights"
        }}
        """
        
        return prompt

    def _parse_llm_tips_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response for tips.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return self._validate_tips_response(parsed)
            else:
                # Fallback to structured parsing
                return self._parse_unstructured_tips_response(response_text)
                
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response, using text parsing")
            return self._parse_unstructured_tips_response(response_text)

    def _validate_tips_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize LLM tips response.
        """
        validated = {
            "tips": [],
            "extraction_confidence": 0.5,
            "analysis_notes": ""
        }
        
        # Validate tips array
        if "tips" in parsed and isinstance(parsed["tips"], list):
            for tip in parsed["tips"]:
                if isinstance(tip, dict):
                    validated_tip = {
                        "title": tip.get("title", "Untitled Tip"),
                        "category": tip.get("category", "process"),
                        "type": tip.get("type", "best_practice"),
                        "priority": max(1, min(4, tip.get("priority", 2))),
                        "description": tip.get("description", ""),
                        "implementation_steps": tip.get("implementation_steps", []),
                        "expected_outcome": tip.get("expected_outcome", ""),
                        "time_to_implement": tip.get("time_to_implement", "Unknown"),
                        "resources_needed": tip.get("resources_needed", []),
                        "confidence": max(0.0, min(1.0, tip.get("confidence", 0.5)))
                    }
                    validated["tips"].append(validated_tip)
        
        # Extract confidence
        if "extraction_confidence" in parsed:
            validated["extraction_confidence"] = max(0.0, min(1.0, parsed["extraction_confidence"]))
        
        # Extract notes
        if "analysis_notes" in parsed:
            validated["analysis_notes"] = parsed["analysis_notes"]
            
        return validated

    def _parse_unstructured_tips_response(self, response: str) -> Dict[str, Any]:
        """
        Parse unstructured LLM response as fallback.
        """
        tips = []
        
        # Look for tip patterns
        tip_sections = re.findall(r'(?:tip|recommendation|suggestion).*?(?=\n\n|\n[A-Z]|$)', 
                                 response, re.IGNORECASE | re.DOTALL)
        
        for i, section in enumerate(tip_sections):
            tip = {
                "title": f"Extracted Tip {i+1}",
                "category": self._extract_tip_category(section),
                "type": "best_practice",
                "priority": 2,
                "description": section.strip(),
                "implementation_steps": self._extract_steps(section),
                "expected_outcome": "Process improvement",
                "time_to_implement": "TBD",
                "resources_needed": [],
                "confidence": 0.4
            }
            tips.append(tip)
        
        return {
            "tips": tips[:10],  # Limit to 10 tips
            "extraction_confidence": 0.4,
            "analysis_notes": "Parsed from unstructured response"
        }

    def _extract_tip_category(self, text: str) -> str:
        """Extract tip category from text."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["automat", "tool", "system"]):
            return "technical"
        elif any(word in text_lower for word in ["communicate", "meeting", "discuss"]):
            return "communication"
        elif any(word in text_lower for word in ["process", "procedure", "workflow"]):
            return "process"
        elif any(word in text_lower for word in ["quality", "standard", "review"]):
            return "quality"
        elif any(word in text_lower for word in ["efficient", "faster", "quick"]):
            return "efficiency"
        else:
            return "productivity"

    def _extract_steps(self, text: str) -> List[str]:
        """Extract implementation steps from text."""
        steps = []
        
        # Look for numbered lists
        numbered_steps = re.findall(r'\d+\.\s*([^\n]+)', text)
        if numbered_steps:
            return numbered_steps[:5]  # Limit to 5 steps
        
        # Look for bullet points
        bullet_steps = re.findall(r'[-•*]\s*([^\n]+)', text)
        if bullet_steps:
            return bullet_steps[:5]
        
        # Fallback: split by common step indicators
        step_indicators = ["first", "second", "then", "next", "finally"]
        for indicator in step_indicators:
            if indicator in text.lower():
                sentences = text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        steps.append(sentence.strip())
        
        return steps[:3] if steps else ["Review and implement as appropriate"]

    def _convert_to_actionable_tips(self, parsed_tips_data: Dict[str, Any]) -> List[ActionableTip]:
        """
        Convert parsed tips data to ActionableTip objects.
        """
        actionable_tips = []
        
        for tip_data in parsed_tips_data.get("tips", []):
            try:
                tip = ActionableTip(
                    title=tip_data.get("title", "Untitled Tip"),
                    category=self._map_to_tip_category(tip_data.get("category", "process")),
                    tip_type=self._map_to_tip_type(tip_data.get("type", "best_practice")),
                    priority=TipPriority(tip_data.get("priority", 2)),
                    description=tip_data.get("description", ""),
                    implementation_steps=tip_data.get("implementation_steps", []),
                    expected_outcome=tip_data.get("expected_outcome", ""),
                    time_to_implement=tip_data.get("time_to_implement", "TBD"),
                    resources_needed=tip_data.get("resources_needed", []),
                    confidence=tip_data.get("confidence", 0.5)
                )
                actionable_tips.append(tip)
                
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Failed to create ActionableTip: {e}")
                continue
        
        return actionable_tips

    def _generate_pattern_tips(self, document: str) -> List[ActionableTip]:
        """
        Generate tips using pattern matching.
        """
        pattern_tips = []
        
        for category, patterns in self.tip_patterns.items():
            for pattern_data in patterns:
                matches = re.finditer(pattern_data["pattern"], document, re.IGNORECASE)
                for match in matches:
                    tip = ActionableTip(
                        title=pattern_data["tip_title"],
                        category=self._map_to_tip_category(category),
                        tip_type=TipType.BEST_PRACTICE,
                        priority=TipPriority(pattern_data.get("priority", 2)),
                        description=pattern_data["description"],
                        implementation_steps=pattern_data.get("steps", ["Review and implement"]),
                        expected_outcome=pattern_data.get("outcome", "Process improvement"),
                        time_to_implement=pattern_data.get("time", "1-2 weeks"),
                        resources_needed=pattern_data.get("resources", ["Team time"]),
                        confidence=0.7
                    )
                    pattern_tips.append(tip)
        
        return pattern_tips[:5]  # Limit pattern tips

    def _combine_and_prioritize_tips(self, llm_tips: List[ActionableTip], 
                                   pattern_tips: List[ActionableTip], 
                                   limit: int) -> List[ActionableTip]:
        """
        Combine and prioritize tips from different sources.
        """
        # Combine all tips
        all_tips = llm_tips + pattern_tips
        
        # Remove duplicates (simple title-based deduplication)
        unique_tips = []
        seen_titles = set()
        
        for tip in all_tips:
            if tip.title.lower() not in seen_titles:
                unique_tips.append(tip)
                seen_titles.add(tip.title.lower())
        
        # Sort by priority (descending) and confidence (descending)
        sorted_tips = sorted(unique_tips, 
                           key=lambda t: (t.priority.value, t.confidence), 
                           reverse=True)
        
        return sorted_tips[:limit]

    def _calculate_overall_confidence(self, tips: List[ActionableTip]) -> float:
        """
        Calculate overall confidence in tip extraction.
        """
        if not tips:
            return 0.0
        
        confidence_sum = sum(tip.confidence for tip in tips)
        base_confidence = confidence_sum / len(tips)
        
        # Adjust based on number of tips and quality indicators
        quality_bonus = 0.0
        if len(tips) >= 5:
            quality_bonus += 0.1
        
        # Bonus for having implementation steps
        tips_with_steps = sum(1 for tip in tips if tip.implementation_steps)
        if tips_with_steps > len(tips) * 0.7:  # 70% have steps
            quality_bonus += 0.1
        
        return min(base_confidence + quality_bonus, 1.0)

    def _generate_summary(self, tips: List[ActionableTip]) -> str:
        """
        Generate summary of extracted tips.
        """
        if not tips:
            return "No actionable tips were extracted from the document."
        
        tip_count = len(tips)
        high_priority = len([t for t in tips if t.priority.value >= 3])
        
        categories = list(set(t.category.value for t in tips))
        primary_category = categories[0] if categories else "general"
        
        summary = f"Extracted {tip_count} actionable tips with {high_priority} high-priority items. "
        summary += f"Primary focus area: {primary_category}. "
        summary += f"Tips span {len(categories)} categories and include specific implementation guidance."
        
        return summary

    def _generate_recommendations(self, tips: List[ActionableTip]) -> List[str]:
        """
        Generate high-level recommendations based on extracted tips.
        """
        recommendations = []
        
        # Priority-based recommendations
        high_priority_tips = [t for t in tips if t.priority.value >= 3]
        if high_priority_tips:
            recommendations.append(f"Prioritize implementation of {len(high_priority_tips)} high-priority improvements")
        
        # Category-based recommendations
        categories = set(t.category.value for t in tips)
        if "automation" in [t.tip_type.value for t in tips]:
            recommendations.append("Consider automation opportunities to improve efficiency")
        
        if "process" in categories:
            recommendations.append("Review and optimize key processes based on identified improvements")
        
        if "communication" in categories:
            recommendations.append("Enhance communication protocols and collaboration methods")
        
        # Time-based recommendations
        quick_wins = [t for t in tips if "day" in t.time_to_implement.lower() or "week" in t.time_to_implement.lower()]
        if quick_wins:
            recommendations.append(f"Start with {len(quick_wins)} quick-win improvements for immediate impact")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _calculate_category_distribution(self, tips: List[ActionableTip]) -> Dict[str, int]:
        """
        Calculate distribution of tips by category.
        """
        distribution = {}
        for tip in tips:
            category = tip.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    # Utility and helper methods
    def _initialize_tip_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize patterns for tip extraction.
        """
        return {
            "automation": [
                {
                    "pattern": r"manual\s+(?:process|task|work)",
                    "tip_title": "Automate Manual Processes",
                    "description": "Identify opportunities to automate repetitive manual tasks",
                    "steps": ["Identify manual processes", "Evaluate automation tools", "Implement automation"],
                    "priority": 3
                }
            ],
            "communication": [
                {
                    "pattern": r"(?:unclear|confusing|ambiguous)",
                    "tip_title": "Improve Communication Clarity",
                    "description": "Enhance clarity in communications and documentation",
                    "steps": ["Review communication standards", "Implement templates", "Provide training"],
                    "priority": 2
                }
            ],
            "process": [
                {
                    "pattern": r"(?:delay|bottleneck|inefficient)",
                    "tip_title": "Optimize Process Flow",
                    "description": "Address process inefficiencies and bottlenecks",
                    "steps": ["Map current process", "Identify bottlenecks", "Redesign workflow"],
                    "priority": 3
                }
            ],
            "quality": [
                {
                    "pattern": r"(?:error|mistake|defect)",
                    "tip_title": "Implement Quality Controls",
                    "description": "Add quality control measures to prevent errors",
                    "steps": ["Analyze error patterns", "Design controls", "Train team"],
                    "priority": 3
                }
            ]
        }

    def _map_to_tip_category(self, category_str: str) -> TipCategory:
        """Map string to TipCategory enum."""
        category_mapping = {
            "productivity": TipCategory.PRODUCTIVITY,
            "communication": TipCategory.COMMUNICATION,
            "technical": TipCategory.TECHNICAL,
            "process": TipCategory.PROCESS,
            "quality": TipCategory.QUALITY,
            "efficiency": TipCategory.EFFICIENCY,
            "collaboration": TipCategory.COLLABORATION,
            "compliance": TipCategory.COMPLIANCE
        }
        
        category_lower = category_str.lower()
        for key, value in category_mapping.items():
            if key in category_lower:
                return value
        
        return TipCategory.PROCESS  # Default

    def _map_to_tip_type(self, type_str: str) -> TipType:
        """Map string to TipType enum."""
        type_mapping = {
            "best_practice": TipType.BEST_PRACTICE,
            "optimization": TipType.OPTIMIZATION,
            "automation": TipType.AUTOMATION,
            "workflow": TipType.WORKFLOW,
            "tool_usage": TipType.TOOL_USAGE,
            "guideline": TipType.GUIDELINE
        }
        
        type_lower = type_str.lower()
        for key, value in type_mapping.items():
            if key in type_lower:
                return value
        
        return TipType.BEST_PRACTICE  # Default

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent."""
        logger = logging.getLogger(f"{__name__}.TipExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _handle_extraction_error(self, error: Exception, document: str) -> TipExtractionResult:
        """
        Handle extraction errors gracefully.
        """
        self.logger.error(f"Tip extraction failed: {str(error)}")
        
        return TipExtractionResult(
            total_tips_extracted=0,
            confidence=0.0,
            summary=f"Tip extraction failed due to error: {str(error)[:100]}",
            recommendations=["Manual review required due to extraction failure"],
            category_distribution={},
            extraction_method="error_fallback",
            tips=[],
            metadata={
                "error": str(error),
                "error_timestamp": datetime.now().isoformat(),
                "document_length": len(document)
            }
        )

    # Additional utility methods
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold dynamically."""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics and performance metrics."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "supported_categories": [cat.value for cat in TipCategory],
            "supported_types": [tip_type.value for tip_type in TipType],
            "tip_patterns_loaded": len(self.tip_patterns),
            "agent_status": "active"
        }