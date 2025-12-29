import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.retrievers import MultiQueryRetriever
from langchain.vectorstores import Chroma
from utils.fallback_rules import RulesetB

class ClauseReviewerAgent:
    def __init__(self, model=None, vectorstore=None, confidence_threshold=0.7):
        self.model = model  # e.g., Gemini 2.5 Pro
        self.vectorstore = vectorstore  # Vector DB with T&C chunks
        self.confidence_threshold = confidence_threshold
        self.fallback_rules = RulesetB()
        
    def review_clauses(self, document: str, context: str = "") -> dict:
        """
        Review clauses using RAG-Fusion approach.
        Falls back to rule-based system if confidence is low.
        """
        try:
            # Step 1: RAG-Fusion retrieval and analysis
            rag_result = self._perform_rag_fusion_analysis(document, context)
            
            # Step 2: Check confidence and decide on fallback
            if rag_result["confidence"] >= self.confidence_threshold:
                return {
                    "analysis": rag_result["analysis"],
                    "method": "rag_fusion",
                    "confidence": rag_result["confidence"],
                    "retrieved_chunks": rag_result["chunks"]
                }
            else:
                # Fallback to Ruleset-B
                fallback_result = self.fallback_rules.analyze_clauses(document)
                return {
                    "analysis": fallback_result["analysis"],
                    "method": "ruleset_fallback",
                    "confidence": fallback_result["confidence"],
                    "triggered_rules": fallback_result["rules_applied"]
                }
                
        except Exception as e:
            # Emergency fallback to rules
            return self._emergency_fallback(document, str(e))

    def _perform_rag_fusion_analysis(self, document: str, context: str) -> dict:
        """
        Core RAG-Fusion implementation with multi-query and RRF.
        """
        # Step 1: Generate multiple queries for clause analysis
        queries = self._generate_multiple_queries(document, context)
        
        # Step 2: Retrieve documents for each query
        all_retrieved_docs = []
        for query in queries:
            docs = self.vectorstore.similarity_search(query, k=5)
            all_retrieved_docs.append(docs)
        
        # Step 3: Apply Reciprocal Rank Fusion (RRF)
        fused_docs = self._reciprocal_rank_fusion(all_retrieved_docs)
        
        # Step 4: Generate analysis using top-ranked documents
        analysis_result = self._generate_clause_analysis(
            document, fused_docs, context
        )
        
        return {
            "analysis": analysis_result["analysis"],
            "confidence": analysis_result["confidence"],
            "chunks": fused_docs[:3]  # Return top 3 chunks
        }

    def _generate_multiple_queries(self, document: str, context: str) -> List[str]:
        """
        Generate multiple perspectives of the clause review query.
        """
        prompt = f"""
        Generate 3-4 diverse queries to analyze clauses in this document:
        
        Document: {document[:500]}...
        Context: {context}
        
        Generate queries covering:
        1. Risk assessment perspective
        2. Compliance perspective  
        3. Contractual obligations perspective
        4. Legal implications perspective
        
        Return queries as a newline-separated list.
        """
        
        response = self.model.generate_text(prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries[:4]  # Limit to 4 queries

    def _reciprocal_rank_fusion(self, all_docs: List[List], k: int = 60) -> List:
        """
        Implement Reciprocal Rank Fusion to combine and rank documents.
        """
        fused_scores = {}
        
        for docs_list in all_docs:
            for rank, doc in enumerate(docs_list):
                doc_id = self._get_doc_identifier(doc)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"doc": doc, "score": 0}
                
                # RRF formula: 1 / (rank + k)
                fused_scores[doc_id]["score"] += 1 / (rank + k)
        
        # Sort by fused score (descending)
        sorted_docs = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return [item["doc"] for item in sorted_docs]

    def _get_doc_identifier(self, doc) -> str:
        """Create unique identifier for document chunks."""
        return f"{doc.metadata.get('source', 'unknown')}_{hash(doc.page_content)}"

    def _generate_clause_analysis(self, document: str, retrieved_docs: List, context: str) -> dict:
        """
        Generate clause analysis using retrieved documents.
        """
        # Combine top retrieved documents
        combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
        
        prompt = f"""
        Analyze the clauses in this document using the provided reference context:
        
        Document to analyze: {document}
        Additional context: {context}
        
        Reference context from T&C database:
        {combined_context}
        
        Provide:
        1. Key clause analysis
        2. Risk assessment (scale 1-10)
        3. Compliance issues identified
        4. Recommendations
        5. Confidence score (0.0-1.0)
        
        Return as JSON format.
        """
        
        response = self.model.generate_text(prompt)
        
        try:
            analysis_json = json.loads(response)
            return {
                "analysis": analysis_json,
                "confidence": analysis_json.get("confidence_score", 0.5)
            }
        except json.JSONDecodeError:
            # Fallback to structured parsing
            return {
                "analysis": {"raw_response": response},
                "confidence": 0.4  # Lower confidence due to parsing issues
            }

    def _emergency_fallback(self, document: str, error: str) -> dict:
        """
        Emergency fallback when RAG-Fusion fails completely.
        """
        try:
            fallback_result = self.fallback_rules.analyze_clauses(document)
            fallback_result["error"] = error
            fallback_result["method"] = "emergency_fallback"
            return fallback_result
        except Exception as fallback_error:
            return {
                "analysis": {"error": "Complete system failure", 
                           "original_error": error,
                           "fallback_error": str(fallback_error)},
                "method": "critical_failure",
                "confidence": 0.0
            }

    def setup_retriever(self, vectorstore_path: str = "./vectorstore/tnc_db.chroma/"):
        """
        Initialize the vector database retriever.
        """
        try:
            from langchain.embeddings import OpenAIEmbeddings  # or your preferred embeddings
            embeddings = OpenAIEmbeddings()
            
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embeddings
            )
            return True
        except Exception as e:
            print(f"Failed to setup retriever: {e}")
            return False

    def update_confidence_threshold(self, new_threshold: float):
        """
        Dynamically adjust confidence threshold.
        """
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")