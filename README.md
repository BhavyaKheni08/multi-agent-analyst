# 🤖 Multi-Agent Document Analyst

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Pro-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Intelligent multi-agent system for comprehensive document analysis using LangGraph orchestration**

Transform complex documents into actionable insights with specialized AI agents working in harmony.

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Configuration](#-configuration)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Agents](#-agents)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## 🎯 Overview

**Multi-Agent Document Analyst** is a sophisticated document analysis system that uses LangGraph to orchestrate multiple specialized AI agents. Each agent focuses on a specific aspect of document analysis, working together to provide comprehensive insights for insurance policies, legal documents, and other complex texts.

### What It Does

- 📄 **Analyzes** complex documents (insurance policies, legal contracts, etc.)
- 📊 **Summarizes** key information and extracts insights
- 🔍 **Reviews** clauses and terms with RAG-powered context
- ⚠️ **Assesses** risks and highlights potential issues
- 💡 **Extracts** actionable tips and recommendations
- 🎯 **Orchestrates** multiple agents using LangGraph state management

### Use Cases

- **Insurance Policy Review**: Comprehensive analysis of policy documents
- **Legal Document Analysis**: Clause review and risk assessment
- **Contract Analysis**: Terms extraction and compliance checking
- **Business Intelligence**: Document insights for decision-making

## ✨ Features

- 🤖 **Four Specialized Agents**: Each agent handles a specific analysis task
- 🔄 **LangGraph Orchestration**: State-based workflow management
- 🧠 **Google Gemini Integration**: Uses Gemini 2.5 Pro for advanced reasoning
- 📚 **RAG-Powered Analysis**: Retrieval Augmented Generation for context-aware reviews
- ⚙️ **Highly Configurable**: Flexible agent configuration and fallback mechanisms
- 📊 **Comprehensive Logging**: Detailed logging system for debugging
- 🎯 **Confidence Scoring**: Configurable confidence thresholds
- 🔁 **Fallback Mechanisms**: Automatic fallback strategies for reliability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestrator                    │
│              (LangGraph StateGraph)                      │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Agent A│ │ Agent B│ │ Agent C│ │ Agent D│
│Summary │ │ Clause │ │  Risk  │ │  Tips  │
│        │ │Reviewer│ │Analyzer│ │Extractor│
└────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
     │          │           │           │
     └──────────┼───────────┼───────────┘
                │           │
                ▼           ▼
         ┌─────────────┐ ┌─────────────┐
         │   Gemini    │ │   Vector    │
         │   2.5 Pro   │ │   Store    │
         └─────────────┘ └─────────────┘
                │
                ▼
         ┌─────────────┐
         │   Final     │
         │  Analysis   │
         └─────────────┘
```

### Agent Responsibilities

1. **Agent A - Summarizer**: Creates concise document summaries
2. **Agent B - Clause Reviewer**: Reviews clauses using RAG with vector store
3. **Agent C - Risk Analyzer**: Identifies risks and potential issues
4. **Agent D - Tip Extractor**: Extracts actionable tips and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Google AI API key (for Gemini)
- (Optional) ChromaDB for vector storage

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-agent-analyst.git
   cd multi-agent-analyst
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Google AI key
   ```

5. **Run the system**
   ```bash
   python main.py --document path/to/document.txt
   ```

## 🤖 Agents

### Agent A: Summarizer

Creates concise summaries of documents with key points extraction.

**Features:**
- Document summarization
- Context-aware summaries
- Fallback modes (concise, detailed, bullet points)
- Confidence scoring

### Agent B: Clause Reviewer

Reviews document clauses using RAG (Retrieval Augmented Generation) with vector store.

**Features:**
- RAG-Fusion for multi-query retrieval
- Vector store integration (ChromaDB)
- Ruleset fallback mechanisms
- Clause-by-clause analysis

### Agent C: Risk Analyzer

Identifies risks, potential issues, and compliance concerns.

**Features:**
- Risk scoring and categorization
- Compliance checking
- Issue highlighting
- Risk mitigation suggestions

### Agent D: Tip Extractor

Extracts actionable tips and recommendations from documents.

**Features:**
- Tip extraction
- Actionable recommendations
- Best practices identification
- Implementation guidance

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# Google AI Configuration (Required)
GOOGLE_AI_KEY=your_google_ai_key_here

# OpenAI Configuration (Optional - backup)
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration (Optional - for vector store)
CHROMA_API_KEY=your_chroma_api_key_here
```

### Configuration File

Edit `config.py` to customize:

```python
# Agent enable/disable
enable_agent_a: bool = True
enable_agent_b: bool = True
enable_agent_c: bool = True
enable_agent_d: bool = True

# Confidence thresholds
min_confidence_threshold: float = 0.4
high_confidence_threshold: float = 0.8

# Retry logic
max_retries: int = 3
timeout_seconds: int = 300

# Fallback mechanisms
enable_fallbacks: bool = True
skip_on_failure: bool = False
```

## 📖 Usage Examples

### Basic Usage

```python
from main import MultiAgentAnalystSystem

# Initialize the system
system = MultiAgentAnalystSystem()

# Analyze a document
with open("document.txt", "r") as f:
    document = f.read()

result = system.analyze_document(
    document=document,
    analysis_type="comprehensive"
)

print(result)
```

### Command Line Usage

```bash
# Analyze a document file
python main.py --document sample_policy.txt

# Use custom configuration
python main.py --document document.txt --config custom_config.json

# Quick analysis mode
python main.py --document document.txt --type quick
```

### Programmatic Usage

```python
import json
from main import MultiAgentAnalystSystem

# Initialize system
system = MultiAgentAnalystSystem()

# Load document
with open("insurance_policy.txt", "r") as f:
    document_content = f.read()

# Run analysis
result = system.analyze_document(
    document=document_content,
    analysis_type="comprehensive",
    user_preferences={
        "focus_areas": ["coverage", "exclusions", "premiums"]
    }
)

# Save results
with open("analysis_result.json", "w") as f:
    json.dump(result, f, indent=2)
```

## 📁 Project Structure

```
multi-agent-analyst/
├── main.py                 # Main entry point and system class
├── orchestrator.py         # LangGraph orchestration logic
├── config.py               # Configuration classes
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
├── agents/                # Agent implementations
│   ├── agent_a.py         # Summarizer agent
│   ├── agent_b.py         # Clause reviewer agent
│   ├── agent_c.py         # Risk analyzer agent
│   └── agent_d.py         # Tip extractor agent
├── utils/                  # Utility functions
│   └── fallback_rules.py  # Fallback rulesets
└── vectorstore/           # Vector database (if using local)
    └── tno_db.chroma/     # ChromaDB vector store
```

## 🔧 Advanced Configuration

### Custom Agent Configuration

```python
from config import AgentAConfig, AgentBConfig

# Customize Agent A
agent_a_config = AgentAConfig(
    enabled=True,
    max_summary_length=1000,
    confidence_threshold=0.7,
    fallback_mode="detailed"
)

# Customize Agent B (RAG)
agent_b_config = AgentBConfig(
    enabled=True,
    multi_query_count=6,
    retrieval_k=10,
    use_ruleset_b_fallback=True
)
```

### LangGraph State Management

The orchestrator uses LangGraph's state management:

```python
class MultiAgentState(TypedDict):
    document: str
    context: str
    summary_result: Optional[Dict[str, Any]]
    clause_analysis_result: Optional[Dict[str, Any]]
    risk_analysis_result: Optional[RiskAnalysisResult]
    tip_extraction_result: Optional[TipExtractionResult]
    final_analysis: Optional[Dict[str, Any]]
    confidence_score: float
    processing_complete: bool
```

## 🧪 Testing

```bash
# Run with sample document
python main.py --document sample_policy.txt

# Test individual agents
python -c "from agents.agent_a import SummarizerAgent; ..."
```

## 📊 Output Format

The system returns structured JSON:

```json
{
  "final_analysis": {
    "summary": "...",
    "clause_review": [...],
    "risk_assessment": {...},
    "tips": [...]
  },
  "confidence_score": 0.85,
  "agents_used": ["agent_a", "agent_b", "agent_c", "agent_d"],
  "processing_time": 12.5,
  "metadata": {...}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration
- [Google Gemini](https://ai.google.dev/) for advanced AI capabilities
- [LangChain](https://www.langchain.com/) for LLM integration

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">

Made with ❤️ by [Your Name]

⭐ Star this repo if you find it helpful!

</div>
