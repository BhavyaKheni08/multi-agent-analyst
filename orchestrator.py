# orchestrator.py

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Literal, Annotated
from datetime import datetime
from dataclasses import dataclass, asdict
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# Agent imports
from agents.agent_a import SummarizerAgent
from agents.agent_b import ClauseReviewerAgent
from agents.agent_c import RiskAnalyzerAgent, RiskAnalysisResult
from agents.agent_d import YouTubeTipExtractorAgent, TipExtractionResult

# Utility imports
from utils.fallback_rules import RulesetB

class ProcessingStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class MultiAgentState(TypedDict):
    """
    Shared state across all agents in the multi-agent system.
    """
    # Input data
    document: str
    context: str
    analysis_type: str
    user_preferences: Dict[str, Any]
    
    # Agent results
    summary_result: Optional[Dict[str, Any]]
    clause_analysis_result: Optional[Dict[str, Any]]
    risk_analysis_result: Optional[RiskAnalysisResult]
    tip_extraction_result: Optional[TipExtractionResult]
    
    # Processing status tracking
    agent_status: Dict[str, str]
    processing_errors: Annotated[List[Dict[str, Any]], operator.add]
    fallback_triggers: Annotated[List[Dict[str, str]], operator.add]
    
    # Final output
    final_analysis: Optional[Dict[str, Any]]
    confidence_score: float
    processing_complete: bool
    
    # Metadata
    session_id: str
    start_time: str
    processing_time: float
    agents_used: List[str]

@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""
    # Agent configurations
    enable_agent_a: bool = True
    enable_agent_b: bool = True
    enable_agent_c: bool = True
    enable_agent_d: bool = True
    
    # Processing controls
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_processing: bool = False
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.4
    high_confidence_threshold: float = 0.8
    
    # Fallback controls
    enable_fallbacks: bool = True
    skip_on_failure: bool = False
    
    # Output preferences
    include_agent_details: bool = True
    include_processing_metrics: bool = True
    consolidate_recommendations: bool = True

class MultiAgentOrchestrator:
    """
    LangGraph-based orchestrator for coordinating multi-agent document analysis.
    """
    
    def __init__(self, config: OrchestratorConfig = None, model=None):
        self.config = config or OrchestratorConfig()
        self.model = model
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with shared model and configurations."""
        agents = {}
        
        try:
            if self.config.enable_agent_a:
                agents['summarizer'] = SummarizerAgent(model=self.model)
                self.logger.info("Initialized Summarizer Agent (A)")
            
            if self.config.enable_agent_b:
                agents['clause_reviewer'] = ClauseReviewerAgent(
                    model=self.model,
                    confidence_threshold=self.config.high_confidence_threshold
                )
                self.logger.info("Initialized Clause Reviewer Agent (B)")
            
            if self.config.enable_agent_c:
                agents['risk_analyzer'] = RiskAnalyzerAgent(
                    model=self.model,
                    confidence_threshold=self.config.high_confidence_threshold,
                    fallback_threshold=self.config.min_confidence_threshold
                )
                self.logger.info("Initialized Risk Analyzer Agent (C)")
            
            if self.config.enable_agent_d:
                agents['tip_extractor'] = YouTubeTipExtractorAgent(
                    model=self.model,
                    confidence_threshold=self.config.min_confidence_threshold
                )
                self.logger.info("Initialized Tip Extractor Agent (D)")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            raise
        
        return agents

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with FSM logic."""
        
        # Create the StateGraph
        workflow = StateGraph(MultiAgentState)
        
        # Add agent nodes
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("agent_a_summarizer", self._run_agent_a)
        workflow.add_node("agent_b_clause_reviewer", self._run_agent_b)
        workflow.add_node("agent_c_risk_analyzer", self._run_agent_c) 
        workflow.add_node("agent_d_tip_extractor", self._run_agent_d)
        workflow.add_node("consolidate_results", self._consolidate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("finalize", self._finalize_processing)
        
        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        
        # Sequential processing flow with conditional edges
        workflow.add_conditional_edges(
            "initialize",
            self._route_after_initialization,
            {
                "agent_a": "agent_a_summarizer",
                "skip_to_consolidate": "consolidate_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_a_summarizer", 
            self._route_after_agent_a,
            {
                "agent_b": "agent_b_clause_reviewer",
                "retry_a": "agent_a_summarizer", 
                "skip_b": "agent_c_risk_analyzer",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_b_clause_reviewer",
            self._route_after_agent_b, 
            {
                "agent_c": "agent_c_risk_analyzer",
                "retry_b": "agent_b_clause_reviewer",
                "skip_c": "agent_d_tip_extractor", 
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_c_risk_analyzer",
            self._route_after_agent_c,
            {
                "agent_d": "agent_d_tip_extractor",
                "retry_c": "agent_c_risk_analyzer",
                "skip_d": "consolidate_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "agent_d_tip_extractor", 
            self._route_after_agent_d,
            {
                "consolidate": "consolidate_results",
                "retry_d": "agent_d_tip_extractor",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("consolidate_results", "finalize")
        workflow.add_edge("handle_error", "finalize")
        workflow.add_edge("finalize", END)
        
        # COMPLETELY DISABLE CHECKPOINTING
        self.logger.info("Compiling workflow without checkpointing")
        return workflow.compile()

    # Node implementations
    
    def _initialize_processing(self, state: MultiAgentState) -> Dict[str, Any]:
        """Initialize processing state and validate inputs."""
        self.logger.info(f"Initializing processing for session: {state.get('session_id', 'unknown')}")
        
        # Validate required inputs
        if not state.get('document'):
            return {
                "processing_errors": [{"error": "No document provided", "timestamp": datetime.now().isoformat()}],
                "processing_complete": False
            }
        
        # Initialize agent status tracking
        agent_status = {}
        for agent_key in ['summarizer', 'clause_reviewer', 'risk_analyzer', 'tip_extractor']:
            if agent_key in self.agents:
                agent_status[agent_key] = ProcessingStatus.PENDING
        
        return {
            "agent_status": agent_status,
            "start_time": datetime.now().isoformat(),
            "processing_complete": False,
            "confidence_score": 0.0,
            "agents_used": [],
            "processing_errors": [],
            "fallback_triggers": []
        }

    def _run_agent_a(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute Agent A (Summarizer) with fallback handling."""
        self.logger.info("Executing Agent A: Summarizer")
        
        try:
            if 'summarizer' not in self.agents:
                return {
                    "agent_status": {**state.get("agent_status", {}), "summarizer": ProcessingStatus.SKIPPED},
                    "processing_errors": [{"agent": "A", "error": "Agent not available", "timestamp": datetime.now().isoformat()}]
                }
            
            # Update status
            updated_status = {**state.get("agent_status", {}), "summarizer": ProcessingStatus.IN_PROGRESS}
            
            # Run summarization
            summarizer = self.agents['summarizer']
            result = summarizer.summarize(
                document=state['document'],
                context=state.get('context', '')
            )
            
            # Check if fallback needed
            if result.get('meta', {}).get('confidence', 1.0) < self.config.min_confidence_threshold:
                self.logger.warning("Agent A confidence low, triggering fallback")
                fallback_result = summarizer.rerun_with_fallback(
                    document=state['document'],
                    context=state.get('context', ''),
                    fallback_mode="concise"
                )
                result = fallback_result
                fallback_triggers = [{"agent": "A", "reason": "low_confidence", "timestamp": datetime.now().isoformat()}]
            else:
                fallback_triggers = []
            
            return {
                "summary_result": result,
                "agent_status": {**updated_status, "summarizer": ProcessingStatus.COMPLETED},
                "agents_used": state.get('agents_used', []) + ['summarizer'],
                "fallback_triggers": fallback_triggers
            }
            
        except Exception as e:
            self.logger.error(f"Agent A failed: {str(e)}")
            return {
                "agent_status": {**state.get("agent_status", {}), "summarizer": ProcessingStatus.FAILED},
                "processing_errors": [{"agent": "A", "error": str(e), "timestamp": datetime.now().isoformat()}]
            }

    def _run_agent_b(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute Agent B (Clause Reviewer) with RAG-Fusion and fallback."""
        self.logger.info("Executing Agent B: Clause Reviewer")
        
        try:
            if 'clause_reviewer' not in self.agents:
                return {
                    "agent_status": {**state.get("agent_status", {}), "clause_reviewer": ProcessingStatus.SKIPPED},
                    "processing_errors": [{"agent": "B", "error": "Agent not available", "timestamp": datetime.now().isoformat()}]
                }
            
            # Update status
            updated_status = {**state.get("agent_status", {}), "clause_reviewer": ProcessingStatus.IN_PROGRESS}
            
            # Use summary from Agent A if available
            context = state.get('context', '')
            if state.get('summary_result'):
                summary_text = state['summary_result'].get('summary', '')
                context = f"{context}\n\nSummary: {summary_text}" if context else f"Summary: {summary_text}"
            
            # Run clause analysis
            clause_reviewer = self.agents['clause_reviewer']
            result = clause_reviewer.review_clauses(
                document=state['document'],
                context=context
            )
            
            # Track fallback usage
            fallback_triggers = []
            if result.get('method') in ['ruleset_fallback', 'emergency_fallback']:
                fallback_triggers = [{"agent": "B", "reason": "rag_fusion_failed", "timestamp": datetime.now().isoformat()}]
            
            return {
                "clause_analysis_result": result,
                "agent_status": {**updated_status, "clause_reviewer": ProcessingStatus.COMPLETED},
                "agents_used": state.get('agents_used', []) + ['clause_reviewer'], 
                "fallback_triggers": fallback_triggers
            }
            
        except Exception as e:
            self.logger.error(f"Agent B failed: {str(e)}")
            return {
                "agent_status": {**state.get("agent_status", {}), "clause_reviewer": ProcessingStatus.FAILED},
                "processing_errors": [{"agent": "B", "error": str(e), "timestamp": datetime.now().isoformat()}]
            }

    def _run_agent_c(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute Agent C (Risk Analyzer) with confidence-based fallback."""
        self.logger.info("Executing Agent C: Risk Analyzer") 
        
        try:
            if 'risk_analyzer' not in self.agents:
                return {
                    "agent_status": {**state.get("agent_status", {}), "risk_analyzer": ProcessingStatus.SKIPPED},
                    "processing_errors": [{"agent": "C", "error": "Agent not available", "timestamp": datetime.now().isoformat()}]
                }
            
            # Update status
            updated_status = {**state.get("agent_status", {}), "risk_analyzer": ProcessingStatus.IN_PROGRESS}
            
            # Build context from previous agents
            context = state.get('context', '')
            
            if state.get('summary_result'):
                summary_text = state['summary_result'].get('summary', '')
                context += f"\n\nDocument Summary: {summary_text}"
            
            if state.get('clause_analysis_result'):
                clause_analysis = state['clause_analysis_result'].get('analysis', {})
                if isinstance(clause_analysis, dict):
                    context += f"\n\nClause Analysis: {json.dumps(clause_analysis, indent=2)}"
            
            # Run risk analysis
            risk_analyzer = self.agents['risk_analyzer']
            result = risk_analyzer.analyze_risks(
                document=state['document'],
                context=context,
                analysis_type=state.get('analysis_type', 'comprehensive')
            )
            
            # Track fallback usage
            fallback_triggers = []
            if result.fallback_used:
                fallback_triggers = [{"agent": "C", "reason": f"method_{result.analysis_method}", "timestamp": datetime.now().isoformat()}]
            
            return {
                "risk_analysis_result": result,
                "agent_status": {**updated_status, "risk_analyzer": ProcessingStatus.COMPLETED},
                "agents_used": state.get('agents_used', []) + ['risk_analyzer'],
                "fallback_triggers": fallback_triggers
            }
            
        except Exception as e:
            self.logger.error(f"Agent C failed: {str(e)}")
            return {
                "agent_status": {**state.get("agent_status", {}), "risk_analyzer": ProcessingStatus.FAILED},
                "processing_errors": [{"agent": "C", "error": str(e), "timestamp": datetime.now().isoformat()}]
            }

    def _run_agent_d(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute Agent D (Tip Extractor) with category mapping."""
        self.logger.info("Executing Agent D: Tip Extractor")
        
        try:
            if 'tip_extractor' not in self.agents:
                return {
                    "agent_status": {**state.get("agent_status", {}), "tip_extractor": ProcessingStatus.SKIPPED},
                    "processing_errors": [{"agent": "D", "error": "Agent not available", "timestamp": datetime.now().isoformat()}]
                }
            
            # Update status  
            updated_status = {**state.get("agent_status", {}), "tip_extractor": ProcessingStatus.IN_PROGRESS}
            
            # Build comprehensive context from all previous agents
            context = state.get('context', '')
            
            if state.get('summary_result'):
                summary_text = state['summary_result'].get('summary', '')
                context += f"\n\nDocument Summary: {summary_text}"
            
            if state.get('clause_analysis_result'):
                clause_analysis = state['clause_analysis_result']
                context += f"\n\nClause Analysis Method: {clause_analysis.get('method', 'unknown')}"
                if 'analysis' in clause_analysis:
                    context += f"\nKey Clause Findings: {str(clause_analysis['analysis'])[:200]}..."
            
            if state.get('risk_analysis_result'):
                risk_result = state['risk_analysis_result']
                context += f"\n\nRisk Analysis: Score {risk_result.overall_risk_score}/10"
                context += f"\nRisk Summary: {risk_result.summary}"
            
            # Run tip extraction
            tip_extractor = self.agents['tip_extractor']
            result = tip_extractor.extract_tips(
                document=state['document'],
                context=context,
                tip_count_limit=15
            )
            
            return {
                "tip_extraction_result": result,
                "agent_status": {**updated_status, "tip_extractor": ProcessingStatus.COMPLETED},
                "agents_used": state.get('agents_used', []) + ['tip_extractor']
            }
            
        except Exception as e:
            self.logger.error(f"Agent D failed: {str(e)}")
            return {
                "agent_status": {**state.get("agent_status", {}), "tip_extractor": ProcessingStatus.FAILED},
                "processing_errors": [{"agent": "D", "error": str(e), "timestamp": datetime.now().isoformat()}]
            }

    def _consolidate_results(self, state: MultiAgentState) -> Dict[str, Any]:
        """Consolidate results from all agents into final analysis."""
        self.logger.info("Consolidating results from all agents")
        
        try:
            # Calculate overall confidence
            confidences = []
            
            if state.get('summary_result'):
                confidences.append(state['summary_result'].get('meta', {}).get('confidence', 0.5))
            
            if state.get('clause_analysis_result'):
                confidences.append(state['clause_analysis_result'].get('confidence', 0.5))
            
            if state.get('risk_analysis_result'):
                confidences.append(state['risk_analysis_result'].confidence)
            
            if state.get('tip_extraction_result'):
                confidences.append(state['tip_extraction_result'].confidence)
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Build final analysis
            final_analysis = {
                "processing_summary": {
                    "agents_executed": len(state.get('agents_used', [])),
                    "agents_used": state.get('agents_used', []),
                    "fallbacks_triggered": len(state.get('fallback_triggers', [])),
                    "errors_encountered": len(state.get('processing_errors', [])),
                    "overall_confidence": overall_confidence
                },
                "document_summary": state.get('summary_result'),
                "clause_analysis": state.get('clause_analysis_result'),
                "risk_analysis": self._serialize_risk_result(state.get('risk_analysis_result')),
                "actionable_tips": self._serialize_tip_result(state.get('tip_extraction_result')),
                "consolidated_recommendations": self._generate_consolidated_recommendations(state)
            }
            
            # Calculate processing time
            start_time = datetime.fromisoformat(state.get('start_time', datetime.now().isoformat()))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "final_analysis": final_analysis,
                "confidence_score": overall_confidence,
                "processing_time": processing_time,
                "processing_complete": True
            }
            
        except Exception as e:
            self.logger.error(f"Result consolidation failed: {str(e)}")
            return {
                "processing_errors": [{"stage": "consolidation", "error": str(e), "timestamp": datetime.now().isoformat()}],
                "processing_complete": False
            }

    def _handle_error(self, state: MultiAgentState) -> Dict[str, Any]:
        """Handle processing errors and decide on recovery actions."""
        self.logger.warning("Handling processing errors")
        
        errors = state.get('processing_errors', [])
        
        # If we have partial results, still try to consolidate
        has_partial_results = any([
            state.get('summary_result'),
            state.get('clause_analysis_result'), 
            state.get('risk_analysis_result'),
            state.get('tip_extraction_result')
        ])
        
        if has_partial_results:
            self.logger.info("Attempting partial result consolidation despite errors")
            partial_analysis = {
                "processing_summary": {
                    "status": "completed_with_errors",
                    "agents_used": state.get('agents_used', []),
                    "error_count": len(errors),
                    "errors": errors[-3:]  # Last 3 errors
                },
                "document_summary": state.get('summary_result'),
                "clause_analysis": state.get('clause_analysis_result'),
                "risk_analysis": self._serialize_risk_result(state.get('risk_analysis_result')),
                "actionable_tips": self._serialize_tip_result(state.get('tip_extraction_result'))
            }
            
            return {
                "final_analysis": partial_analysis,
                "confidence_score": 0.3,  # Low confidence due to errors
                "processing_complete": True
            }
        else:
            # Complete failure
            return {
                "final_analysis": {
                    "error": "Complete processing failure",
                    "errors": errors,
                    "status": "failed"
                },
                "confidence_score": 0.0,
                "processing_complete": True
            }

    def _finalize_processing(self, state: MultiAgentState) -> Dict[str, Any]:
        """Finalize processing and prepare output."""
        self.logger.info("Finalizing processing")
        
        # Log final statistics
        processing_time = state.get('processing_time', 0)
        agents_used = state.get('agents_used', [])
        confidence = state.get('confidence_score', 0)
        
        self.logger.info(f"Processing completed in {processing_time:.2f}s")
        self.logger.info(f"Agents used: {', '.join(agents_used)}")
        self.logger.info(f"Overall confidence: {confidence:.2f}")
        
        return {
            "processing_complete": True
        }

    # Routing functions (conditional edge logic)
    
    def _route_after_initialization(self, state: MultiAgentState) -> Literal["agent_a", "skip_to_consolidate", "error"]:
        """Route after initialization based on state."""
        if state.get('processing_errors'):
            return "error"
        elif 'summarizer' in self.agents:
            return "agent_a"
        else:
            return "skip_to_consolidate"

    def _route_after_agent_a(self, state: MultiAgentState) -> Literal["agent_b", "retry_a", "skip_b", "error"]:
        """Route after Agent A based on results."""
        agent_status = state.get('agent_status', {})
        
        if agent_status.get('summarizer') == ProcessingStatus.FAILED:
            if self._should_retry('summarizer', state):
                return "retry_a"
            elif 'clause_reviewer' in self.agents:
                return "agent_b"  # Continue despite failure
            else:
                return "skip_b"
        elif agent_status.get('summarizer') == ProcessingStatus.COMPLETED:
            if 'clause_reviewer' in self.agents:
                return "agent_b"
            else:
                return "skip_b"
        else:
            return "error"

    def _route_after_agent_b(self, state: MultiAgentState) -> Literal["agent_c", "retry_b", "skip_c", "error"]:
        """Route after Agent B based on results."""
        agent_status = state.get('agent_status', {})
        
        if agent_status.get('clause_reviewer') == ProcessingStatus.FAILED:
            if self._should_retry('clause_reviewer', state):
                return "retry_b"
            elif 'risk_analyzer' in self.agents:
                return "agent_c"
            else:
                return "skip_c"
        elif agent_status.get('clause_reviewer') == ProcessingStatus.COMPLETED:
            if 'risk_analyzer' in self.agents:
                return "agent_c"
            else:
                return "skip_c"
        else:
            return "error"

    def _route_after_agent_c(self, state: MultiAgentState) -> Literal["agent_d", "retry_c", "skip_d", "error"]:
        """Route after Agent C based on results."""
        agent_status = state.get('agent_status', {})
        
        if agent_status.get('risk_analyzer') == ProcessingStatus.FAILED:
            if self._should_retry('risk_analyzer', state):
                return "retry_c"
            elif 'tip_extractor' in self.agents:
                return "agent_d"
            else:
                return "skip_d"
        elif agent_status.get('risk_analyzer') == ProcessingStatus.COMPLETED:
            if 'tip_extractor' in self.agents:
                return "agent_d"
            else:
                return "skip_d"
        else:
            return "error"

    def _route_after_agent_d(self, state: MultiAgentState) -> Literal["consolidate", "retry_d", "error"]:
        """Route after Agent D based on results."""
        agent_status = state.get('agent_status', {})
        
        if agent_status.get('tip_extractor') == ProcessingStatus.FAILED:
            if self._should_retry('tip_extractor', state):
                return "retry_d"
            else:
                return "consolidate"  # Continue to consolidation even if tip extraction failed
        elif agent_status.get('tip_extractor') == ProcessingStatus.COMPLETED:
            return "consolidate"
        else:
            return "error"

    # Utility methods
    
    def _should_retry(self, agent_name: str, state: MultiAgentState) -> bool:
        """Determine if an agent should be retried based on failure patterns."""
        errors = state.get('processing_errors', [])
        agent_errors = [e for e in errors if e.get('agent') == agent_name.replace('_', ' ').title()]
        
        # Simple retry logic - don't retry more than max_retries times
        return len(agent_errors) < self.config.max_retries

    def _serialize_risk_result(self, risk_result: Optional[RiskAnalysisResult]) -> Optional[Dict[str, Any]]:
        """Serialize RiskAnalysisResult for JSON output."""
        if not risk_result:
            return None
        
        return {
            "overall_risk_score": risk_result.overall_risk_score,
            "confidence": risk_result.confidence,
            "summary": risk_result.summary,
            "recommendations": risk_result.recommendations,
            "fallback_used": risk_result.fallback_used,
            "analysis_method": risk_result.analysis_method,
            "risk_tags": [
                {
                    "category": tag.category.value,
                    "severity": tag.severity.value,
                    "confidence": tag.confidence,
                    "keywords": tag.keywords,
                    "context": tag.context[:200] + "..." if len(tag.context) > 200 else tag.context,
                    "mitigation_suggested": tag.mitigation_suggested
                }
                for tag in risk_result.risk_tags
            ],
            "metadata": risk_result.metadata
        }

    def _serialize_tip_result(self, tip_result: Optional[TipExtractionResult]) -> Optional[Dict[str, Any]]:
        """Serialize TipExtractionResult for JSON output."""
        if not tip_result:
            return None
        
        return {
            "total_tips_extracted": tip_result.total_tips_extracted,
            "confidence": tip_result.confidence,
            "summary": tip_result.summary,
            "recommendations": tip_result.recommendations,
            "category_distribution": tip_result.category_distribution,
            "extraction_method": tip_result.extraction_method,
            "tips": [
                {
                    "title": tip.title,
                    "category": tip.category.value,
                    "tip_type": tip.tip_type.value,
                    "priority": tip.priority.value,
                    "description": tip.description,
                    "implementation_steps": tip.implementation_steps,
                    "expected_outcome": tip.expected_outcome,
                    "time_to_implement": tip.time_to_implement,
                    "resources_needed": tip.resources_needed,
                    "confidence": tip.confidence
                }
                for tip in tip_result.tips
            ],
            "metadata": tip_result.metadata
        }

    def _generate_consolidated_recommendations(self, state: MultiAgentState) -> List[str]:
        """Generate consolidated recommendations from all agents."""
        recommendations = []
        
        # From summary
        if state.get('summary_result'):
            summary_recs = state['summary_result'].get('recommendations', [])
            recommendations.extend(summary_recs)
        
        # From clause analysis
        if state.get('clause_analysis_result'):
            clause_recs = state['clause_analysis_result'].get('analysis', {}).get('recommendations', [])
            if isinstance(clause_recs, list):
                recommendations.extend(clause_recs)
        
        # From risk analysis
        if state.get('risk_analysis_result'):
            risk_recs = state['risk_analysis_result'].recommendations
            recommendations.extend(risk_recs)
        
        # From tip extraction
        if state.get('tip_extraction_result'):
            tip_recs = state['tip_extraction_result'].recommendations
            recommendations.extend(tip_recs)
        
        # Deduplicate and prioritize
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the orchestrator."""
        logger = logging.getLogger(f"{__name__}.MultiAgentOrchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    # Public interface methods
    
    def process_document(self, 
                        document: str, 
                        context: str = "",
                        analysis_type: str = "comprehensive",
                        session_id: str = None,
                        user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing documents through the multi-agent system.
        
        Args:
            document (str): Document to analyze
            context (str): Additional context
            analysis_type (str): Type of analysis ('comprehensive', 'quick', 'focused')
            session_id (str): Session identifier for tracking
            user_preferences (Dict): User preferences for processing
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state = MultiAgentState(
            document=document,
            context=context,
            analysis_type=analysis_type,
            user_preferences=user_preferences or {},
            summary_result=None,
            clause_analysis_result=None,
            risk_analysis_result=None,
            tip_extraction_result=None,
            agent_status={},
            processing_errors=[],
            fallback_triggers=[],
            final_analysis=None,
            confidence_score=0.0,
            processing_complete=False,
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            processing_time=0.0,
            agents_used=[]
        )
        
        try:
            # Execute the workflow - COMPLETELY FIXED CHECKPOINTER ISSUE
            self.logger.info(f"Starting document processing for session: {session_id}")
            
            # Direct invoke without any checkpointer configuration
            final_state = self.workflow.invoke(initial_state)
            
            # Extract final analysis from the state
            final_analysis = final_state.get('final_analysis')
            if final_analysis:
                return final_analysis
            else:
                # If no final_analysis, build it from individual components
                return {
                    "processing_summary": {
                        "agents_executed": len(final_state.get('agents_used', [])),
                        "agents_used": final_state.get('agents_used', []),
                        "overall_confidence": final_state.get('confidence_score', 0.0),
                        "processing_time": final_state.get('processing_time', 0.0)
                    },
                    "document_summary": final_state.get('summary_result'),
                    "clause_analysis": final_state.get('clause_analysis_result'),
                    "risk_analysis": self._serialize_risk_result(final_state.get('risk_analysis_result')),
                    "actionable_tips": self._serialize_tip_result(final_state.get('tip_extraction_result')),
                    "session_id": session_id
                }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "error": f"Workflow execution failed: {str(e)}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

    def get_workflow_visualization(self) -> bytes:
        """Get a visual representation of the workflow graph."""
        try:
            return self.workflow.get_graph().draw_mermaid_png()
        except Exception as e:
            self.logger.error(f"Failed to generate workflow visualization: {str(e)}")
            return None

    def update_config(self, new_config: OrchestratorConfig):
        """Update orchestrator configuration and rebuild workflow if needed."""
        self.config = new_config
        self.agents = self._initialize_agents()
        self.workflow = self._build_workflow()
        self.logger.info("Orchestrator configuration updated")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents."""
        return {
            "available_agents": list(self.agents.keys()),
            "config": asdict(self.config),
            "workflow_ready": self.workflow is not None
        }

# Factory function for easy instantiation
def create_orchestrator(config: OrchestratorConfig = None, model=None) -> MultiAgentOrchestrator:
    """
    Factory function to create and configure the multi-agent orchestrator.
    
    Args:
        config (OrchestratorConfig): Configuration for the orchestrator
        model: LLM model to use across all agents
        
    Returns:
        MultiAgentOrchestrator: Configured orchestrator instance
    """
    return MultiAgentOrchestrator(config=config, model=model)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = OrchestratorConfig(
        enable_agent_a=True,
        enable_agent_b=True,
        enable_agent_c=True,
        enable_agent_d=True,
        max_retries=2,
        timeout_seconds=300,
        min_confidence_threshold=0.5,
        high_confidence_threshold=0.8,
        include_agent_details=True,
        consolidate_recommendations=True
    )
    
    # Create orchestrator (model would be injected in real usage)
    orchestrator = create_orchestrator(config=config, model=None)
    
    # Example document processing
    sample_document = """
    This is a sample contract document that needs analysis.
    It contains various clauses, terms, and conditions that require review.
    There may be risks and opportunities for improvement.
    """
    
    # Process document
    try:
        result = orchestrator.process_document(
            document=sample_document,
            context="Sample contract analysis",
            analysis_type="comprehensive"
        )
        
        print("Processing completed successfully!")
        print(f"Results: {json.dumps(result, indent=2, default=str)}")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")