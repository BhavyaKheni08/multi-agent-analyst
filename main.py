import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import configuration and orchestrator
from config import (
    APIConfig, AgentAConfig, AgentBConfig, AgentCConfig, AgentDConfig,
    OrchestratorConfig, SystemConfig
)
from orchestrator import MultiAgentOrchestrator, OrchestratorConfig as GraphConfig

# Google AI Studio model interface
import google.generativeai as genai

class MultiAgentAnalystSystem:
    def __init__(self, config_file: str = None):
        self.config = self._load_configuration(config_file)
        self.logger = self._setup_logging()
        self.model = None
        self.orchestrator = None
        self._initialize_system()

    def _load_configuration(self, config_file: str = None) -> Dict[str, Any]:
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                self.logger.info(f"Loaded custom configuration from {config_file}")
                return custom_config
            except Exception as e:
                print(f"Failed to load config file {config_file}: {e}")
                print("Using default configuration...")
        return {
            'api': APIConfig(),
            'agent_a': AgentAConfig(),
            'agent_b': AgentBConfig(),
            'agent_c': AgentCConfig(),
            'agent_d': AgentDConfig(),
            'orchestrator': OrchestratorConfig(),
            'system': SystemConfig()
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("MultiAgentAnalyst")
        log_level = getattr(logging, self.config.get('system', SystemConfig()).log_level.value)
        logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"multi_agent_analyst_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        return logger

    def _initialize_system(self):
        try:
            self.model = self._initialize_google_ai_model()
            orchestrator_config = GraphConfig(
                enable_agent_a=self.config['agent_a'].enabled,
                enable_agent_b=self.config['agent_b'].enabled,
                enable_agent_c=self.config['agent_c'].enabled,
                enable_agent_d=self.config['agent_d'].enabled,
                max_retries=self.config['orchestrator'].max_retries,
                timeout_seconds=self.config['orchestrator'].timeout_seconds,
                min_confidence_threshold=self.config['orchestrator'].min_confidence_threshold,
                high_confidence_threshold=self.config['orchestrator'].high_confidence_threshold,
                enable_fallbacks=self.config['orchestrator'].enable_fallbacks
            )
            self.orchestrator = MultiAgentOrchestrator(
                config=orchestrator_config,
                model=self.model
            )
            self.logger.info("Multi-agent system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

    def _initialize_google_ai_model(self):
        api_config = self.config['api']
        if not api_config.google_ai_key:
            raise ValueError("Google AI API key not found. Please set GOOGLE_AI_KEY environment variable.")
        genai.configure(api_key=api_config.google_ai_key)
        try:
            model = genai.GenerativeModel(
                model_name=api_config.gemini_model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=api_config.max_tokens_per_request,
                    top_p=0.8,
                    top_k=40
                )
            )
            self.logger.info(f"Initialized {api_config.gemini_model_name} model")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize Google AI model: {str(e)}")
            raise

    def process_document(self, document: str, context: str = "", analysis_type: str = "comprehensive", output_file: str = None) -> Dict[str, Any]:
        if not self.orchestrator:
            raise RuntimeError("System not properly initialized")
        self.logger.info("Starting document processing...")
        self.logger.info(f"Document length: {len(document)} characters")
        self.logger.info(f"Analysis type: {analysis_type}")
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            results = self.orchestrator.process_document(
                document=document,
                context=context,
                analysis_type=analysis_type,
                session_id=session_id,
                user_preferences=self._get_user_preferences()
            )
            # Add system metadata
            results['system_metadata'] = {
                'session_id': session_id,
                'processing_timestamp': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'model_used': self.config['api'].gemini_model_name,
                'configuration': {
                    'agents_enabled': {
                        'summarizer': self.config['agent_a'].enabled,
                        'clause_reviewer': self.config['agent_b'].enabled,
                        'risk_analyzer': self.config['agent_c'].enabled,
                        'tip_extractor': self.config['agent_d'].enabled
                    }
                }
            }
            if output_file:
                self._save_results(results, output_file)
            self.logger.info("Document processing completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise

    def _get_user_preferences(self) -> Dict[str, Any]:
        return {
            'include_agent_details': self.config['orchestrator'].include_agent_details,
            'include_processing_metrics': self.config['orchestrator'].include_processing_metrics,
            'consolidate_recommendations': self.config['orchestrator'].consolidate_recommendations,
            'max_recommendations': 10,
            'preferred_output_format': 'json'
        }

    def _save_results(self, results: Dict[str, Any], output_file: str):
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    def run_health_check(self) -> bool:
        """Run a basic health check of the system."""
        try:
            self.logger.info("Running health check...")
            if not self.model:
                self.logger.error("Model not initialized")
                return False
            if not self.orchestrator:
                self.logger.error("Orchestrator not initialized")
                return False
            self.logger.info("Health check passed")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "system_ready": self.model is not None and self.orchestrator is not None,
            "model_initialized": self.model is not None,
            "orchestrator_initialized": self.orchestrator is not None,
            "agents_enabled": {
                'summarizer': self.config['agent_a'].enabled,
                'clause_reviewer': self.config['agent_b'].enabled,
                'risk_analyzer': self.config['agent_c'].enabled,
                'tip_extractor': self.config['agent_d'].enabled
            },
            "configuration": {
                "model": self.config['api'].gemini_model_name,
                "max_retries": self.config['orchestrator'].max_retries,
                "timeout": self.config['orchestrator'].timeout_seconds
            }
        }

def debug_results_structure(results: Dict[str, Any]):
    """Debug function to show the actual structure of results."""
    print("\n=== DEBUG: RESULTS STRUCTURE ===")
    print("Top-level keys:", list(results.keys()))
    
    # Check for final_analysis
    if 'final_analysis' in results:
        final_analysis = results['final_analysis']
        print("final_analysis keys:", list(final_analysis.keys()) if isinstance(final_analysis, dict) else type(final_analysis))
        
        if isinstance(final_analysis, dict) and 'processing_summary' in final_analysis:
            processing_summary = final_analysis['processing_summary']
            print("processing_summary keys:", list(processing_summary.keys()) if isinstance(processing_summary, dict) else type(processing_summary))
            print("processing_summary content:", processing_summary)
    
    # Check for other possible locations
    if 'processing_summary' in results:
        print("Direct processing_summary:", results['processing_summary'])
    
    if 'overall_confidence' in results:
        print("Direct overall_confidence:", results['overall_confidence'])
    
    if 'agents_used' in results:
        print("Direct agents_used:", results['agents_used'])
    
    print("=== END DEBUG ===\n")

def print_results_summary(results: Dict[str, Any]):
    """Print a clean summary of processing results with debugging."""
    # First, debug the structure
    debug_results_structure(results)
    
    print("=== PROCESSING COMPLETED ===")
    
    # Try multiple possible locations for the data
    confidence = None
    agents_used = []
    processing_time = None
    
    # Method 1: Check final_analysis.processing_summary (expected)
    processing_summary = results.get("final_analysis", {}).get("processing_summary", {})
    if processing_summary:
        confidence = processing_summary.get("overall_confidence")
        agents_used = processing_summary.get("agents_used", [])
        processing_time = processing_summary.get("processing_time")
    
    # Method 2: Check direct keys as fallback
    if confidence is None:
        confidence = results.get("overall_confidence")
    if not agents_used:
        agents_used = results.get("agents_used", [])
    if processing_time is None:
        processing_time = results.get("processing_time")
    
    # Method 3: Check other possible nested locations
    if confidence is None:
        confidence = results.get("processing_summary", {}).get("overall_confidence")
    if not agents_used:
        agents_used = results.get("processing_summary", {}).get("agents_used", [])

    # Print confidence score
    if isinstance(confidence, (float, int)):
        print(f"Confidence Score: {confidence:.2f}")
    else:
        print(f"Confidence Score: {confidence if confidence is not None else 'N/A'}")

    # Print agents used
    if agents_used and isinstance(agents_used, list):
        print(f"Agents Used: {', '.join(agents_used)}")
    else:
        print("Agents Used: None")

    # Print processing time
    if isinstance(processing_time, (float, int)):
        print(f"Processing Time: {processing_time:.2f}s")

    # Print tips generated
    tips = results.get("final_analysis", {}).get("actionable_tips", {})
    if not tips:
        tips = results.get("actionable_tips", {})
    if tips and 'total_tips_extracted' in tips:
        print(f"Tips Generated: {tips['total_tips_extracted']}")

    # Print Document Summary from Agent A
    summary = results.get("final_analysis", {}).get("document_summary", {}).get("summary")
    if not summary:
        summary = results.get("document_summary", {}).get("summary")
    if not summary:
        # Try other possible locations
        summary_result = results.get("final_analysis", {}).get("document_summary")
        if summary_result and isinstance(summary_result, dict):
            summary = summary_result.get("summary") or summary_result.get("summary_text")
    
    if summary:
        print("\n=== Document Summary ===")
        print(summary)
    else:
        print("\nDocument Summary: None")

    # Print Risk Analysis Summary if available
    risk_analysis = results.get("final_analysis", {}).get("risk_analysis", {})
    if not risk_analysis:
        risk_analysis = results.get("risk_analysis", {})
    
    if risk_analysis and 'overall_risk_score' in risk_analysis:
        print(f"\nRisk Score: {risk_analysis['overall_risk_score']:.1f}/10")
        if 'summary' in risk_analysis:
            print(f"Risk Summary: {risk_analysis['summary']}")

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Document Analyst System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file contract.txt --type comprehensive --output results.json
  python main.py --text "Contract content here..." --type quick
  python main.py --health-check
  python main.py --interactive
        """
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--file', '-f', type=str, help='Path to input document file')
    group.add_argument('--text', '-t', type=str, help='Direct input text to analyze')
    parser.add_argument('--context', '-c', type=str, default='', help='Optional additional context')
    parser.add_argument('--type', choices=['comprehensive', 'quick', 'focused'], default='comprehensive', help='Analysis type')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--health-check', action='store_true', help='Perform a system health check')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output except errors')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser

def interactive_mode(system: MultiAgentAnalystSystem):
    print("\nMulti-Agent Document Analyst - Interactive Mode")
    while True:
        print("\nOptions:")
        print("1. Analyze text input")
        print("2. Analyze file")
        print("3. Show system status")
        print("4. Run health check")
        print("5. Exit")
        choice = input("Select option [1-5]: ").strip()
        
        if choice == '1':
            print("Enter document text (Ctrl-D to finish):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            doc_text = "\n".join(lines).strip()
            if doc_text:
                context = input("Additional context (optional): ").strip()
                analysis_type = input("Analysis type (comprehensive/quick/focused) [comprehensive]: ").strip() or "comprehensive"
                try:
                    results = system.process_document(doc_text, context, analysis_type)
                    print_results_summary(results)
                except Exception as e:
                    print(f"Error: {e}")
                    
        elif choice == '2':
            file_path = input("Enter file path: ").strip()
            if file_path:
                context = input("Additional context (optional): ").strip()
                analysis_type = input("Analysis type (comprehensive/quick/focused) [comprehensive]: ").strip() or "comprehensive"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = f.read()
                    results = system.process_document(doc, context, analysis_type)
                    print_results_summary(results)
                except Exception as e:
                    print(f"Error: {e}")
                    
        elif choice == '3':
            status = system.get_system_status()
            print("\n=== System Status ===")
            print(json.dumps(status, indent=2, default=str))
            
        elif choice == '4':
            print("\nRunning system health check...")
            healthy = system.run_health_check()
            print(f"Health check {'PASSED' if healthy else 'FAILED'}")
            
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid option, please select 1-5.")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        print("Initializing Multi-Agent Document Analyst System...")
        system = MultiAgentAnalystSystem(config_file=args.config)
        print("✓ System initialized successfully")

        if args.health_check:
            healthy = system.run_health_check()
            sys.exit(0 if healthy else 1)
            
        elif args.status:
            status = system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)
            
        elif args.interactive:
            interactive_mode(system)
            sys.exit(0)
            
        elif args.file:
            print(f"\nProcessing file: {args.file}")
            with open(args.file, 'r', encoding='utf-8') as f:
                document = f.read()
            results = system.process_document(
                document=document,
                context=args.context,
                analysis_type=args.type,
                output_file=args.output
            )
            
            if not args.quiet:
                print_results_summary(results)
                if args.output:
                    print(f"\nFull results saved to: {args.output}")
                    
        elif args.text:
            print("\nProcessing text input...")
            results = system.process_document(
                document=args.text,
                context=args.context,
                analysis_type=args.type,
                output_file=args.output
            )
            
            if not args.quiet:
                print_results_summary(results)
                if args.output:
                    print(f"\nFull results saved to: {args.output}")
                    
        else:
            print("No input provided. Use --help for usage information or --interactive for interactive mode.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()