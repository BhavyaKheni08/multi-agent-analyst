# Usage Examples

This document provides practical examples of using the Multi-Agent Document Analyst.

## Basic Usage

### Command Line

```bash
# Analyze a document file
python main.py --document sample_policy.txt

# Specify analysis type
python main.py --document document.txt --type comprehensive

# Use custom configuration file
python main.py --document document.txt --config custom_config.json
```

### Python API

```python
from main import MultiAgentAnalystSystem

# Initialize the system
system = MultiAgentAnalystSystem()

# Load document
with open("insurance_policy.txt", "r") as f:
    document = f.read()

# Run comprehensive analysis
result = system.analyze_document(
    document=document,
    analysis_type="comprehensive"
)

# Access results
print(f"Summary: {result['final_analysis']['summary']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Risks: {result['final_analysis']['risk_assessment']}")
```

## Advanced Examples

### Custom Agent Configuration

```python
from main import MultiAgentAnalystSystem
from config import AgentAConfig, AgentBConfig, OrchestratorConfig

# Create custom configuration
custom_config = {
    'agent_a': AgentAConfig(
        enabled=True,
        max_summary_length=1000,
        confidence_threshold=0.7
    ),
    'agent_b': AgentBConfig(
        enabled=True,
        multi_query_count=6,
        retrieval_k=10
    ),
    'orchestrator': OrchestratorConfig(
        enable_agent_c=False,  # Disable risk analyzer
        min_confidence_threshold=0.5
    )
}

system = MultiAgentAnalystSystem(config_file=None)
system.config.update(custom_config)

result = system.analyze_document(document)
```

### Focused Analysis

```python
# Analyze only specific aspects
result = system.analyze_document(
    document=document,
    analysis_type="focused",
    user_preferences={
        "focus_areas": ["coverage", "exclusions", "premiums"],
        "skip_summary": False
    }
)
```

### Error Handling

```python
try:
    result = system.analyze_document(document)
except Exception as e:
    print(f"Analysis failed: {e}")
    # Check logs for details
    # system.logger.error("Analysis error", exc_info=True)
```

## Output Examples

### Comprehensive Analysis Result

```json
{
  "final_analysis": {
    "summary": "This insurance policy provides comprehensive coverage...",
    "clause_review": [
      {
        "clause": "Section 3.2",
        "review": "This clause clearly defines coverage limits...",
        "confidence": 0.85
      }
    ],
    "risk_assessment": {
      "overall_risk": "low",
      "risks": [
        {
          "type": "coverage_gap",
          "severity": "medium",
          "description": "Gap in coverage for natural disasters"
        }
      ]
    },
    "tips": [
      "Consider adding riders for additional coverage",
      "Review premium payment terms carefully"
    ]
  },
  "confidence_score": 0.82,
  "agents_used": ["agent_a", "agent_b", "agent_c", "agent_d"],
  "processing_time": 15.3,
  "metadata": {
    "session_id": "abc123",
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

## Integration Examples

### Flask API Integration

```python
from flask import Flask, request, jsonify
from main import MultiAgentAnalystSystem

app = Flask(__name__)
system = MultiAgentAnalystSystem()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    document = data.get('document')
    analysis_type = data.get('type', 'comprehensive')
    
    result = system.analyze_document(
        document=document,
        analysis_type=analysis_type
    )
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Batch Processing

```python
import asyncio
from main import MultiAgentAnalystSystem

async def batch_analyze(documents):
    system = MultiAgentAnalystSystem()
    results = []
    
    for doc in documents:
        result = system.analyze_document(doc)
        results.append(result)
    
    return results

# Usage
documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
results = asyncio.run(batch_analyze(documents))
```

