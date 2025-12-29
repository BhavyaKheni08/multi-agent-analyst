from typing import Dict, Any, Optional

class SummarizerAgent:
    def __init__(self, model=None):
        self.model = model  # Should be a google.generativeai.GenerativeModel object

    def summarize(self, document: str, context: str = "") -> Dict[str, Any]:
        """
        Summarize a given document, optionally using additional context.
        Returns a dict: {"summary": str, "meta": dict}
        """
        if not self.model:
            raise ValueError("No model specified for summarization.")

        prompt = self._format_prompt(document, context)

        # Use the correct Gemini call
        response = self.model.generate_content(prompt)
        summary = response.text

        # Optionally, set a basic confidence
        meta = {
            "length": len(summary),
            "confidence": 1.0 if summary and len(summary) > 60 else 0.4
        }
        return {"summary": summary, "meta": meta}

    def _format_prompt(self, document, context):
        prompt = f"Summarize the following document:\n{document}\n"
        if context:
            prompt += f"\nContext:\n{context}\n"
        prompt += "\nGive a concise summary and extract key points."
        return prompt

    def _parse_response(self, response):
        """Not needed if you just want response.text. Left for extensibility."""
        return response.text, {"length": len(response.text)}

    def rerun_with_fallback(self, document, context, fallback_mode="concise"):
        """
        Rerun summarization in fallback mode (e.g., only top 3 points).
        """
        prompt = self._format_prompt(document, context)
        if fallback_mode == "concise":
            prompt += "\nIf the previous summary was unclear, just focus on the 3 most important points in a very concise way."
        response = self.model.generate_content(prompt)
        summary = response.text
        meta = {
            "length": len(summary),
            "confidence": 1.0 if summary and len(summary) > 30 else 0.3,
            "fallback": True,
            "fallback_mode": fallback_mode
        }
        return {"summary": summary, "meta": meta}
