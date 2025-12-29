import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class SystemConfig:
    log_level: LogLevel = LogLevel.INFO
    output_dir: str = "output"
    log_dir: str = "logs"
    version: str = "1.0.0"

class AnalysisType(Enum):
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"
    FOCUSED = "focused"

@dataclass
class OrchestratorConfig:
    enable_agent_a: bool = True
    enable_agent_b: bool = True
    enable_agent_c: bool = True
    enable_agent_d: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_processing: bool = False
    min_confidence_threshold: float = 0.4
    high_confidence_threshold: float = 0.8
    enable_fallbacks: bool = True
    skip_on_failure: bool = False
    include_agent_details: bool = True
    include_processing_metrics: bool = True
    consolidate_recommendations: bool = True

@dataclass
class APIConfig:
    """
    API keys and credentials for external services and LLM providers.
    """
    # Google AI Studio configuration
    google_ai_key: str = os.getenv("GOOGLE_AI_KEY", "")
    gemini_model_name: str = "gemini-2.5-pro"  # Default model as discussed
    
    # OpenAI configuration (backup/alternative)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model_name: str = "Qwen"
    
    # Vector DB configuration
    chroma_api_key: str = os.getenv("CHROMA_API_KEY", "")
    embedding_model: str = "llama-text-embed-v2"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_request: int = 8000

@dataclass
class AgentAConfig:
    """
    Configuration for Agent A - Summarizer (Document + Context).
    """
    enabled: bool = True
    fallback_mode: str = "concise"  # Options: "concise", "detailed", "bullet_points"
    max_summary_length: int = 500
    confidence_threshold: float = 0.5
    retry_attempts: int = 2
    timeout_seconds: int = 60

@dataclass
class AgentBConfig:
    """
    Configuration for Agent B - Clause Reviewer (RAG-Fusion).
    """
    enabled: bool = True
    confidence_threshold: float = 0.7
    fallback_to_ruleset: bool = True
    
    # RAG-Fusion parameters
    multi_query_count: int = 4
    retrieval_k: int = 5
    rrf_k_value: int = 60  # Reciprocal Rank Fusion parameter
    
    # Vector database settings
    vectorstore_path: str = "./vectorstore/tnc_db.chroma/"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Fallback settings
    use_ruleset_b_fallback: bool = True
    emergency_fallback: bool = True
    retry_attempts: int = 2
    timeout_seconds: int = 120

@dataclass
class AgentCConfig:
    """
    Configuration for Agent C - Risk Analyzer (LLM + Tagging).
    """
    default_analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE    
    enabled: bool = True
    confidence_threshold: float = 0.6
    fallback_threshold: float = 0.4

@dataclass
class AgentDConfig:
    """
    Configuration for Agent D - YouTube Tip Extractor.
    """
    enabled: bool = True
    max_tips_per_category: int = 5
    confidence_threshold: float = 0.6
    tip_count_limit: int = 15
    allow_pattern_fallback: bool = True
    retry_attempts: int = 2
    timeout_seconds: int = 90