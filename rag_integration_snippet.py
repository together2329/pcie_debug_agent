
# Add this to your interactive shell initialization

try:
    from enhanced_context_rag import integrate_contextual_rag
    integrate_contextual_rag(self)
    print("✅ Enhanced contextual RAG integrated")
except Exception as e:
    print(f"⚠️  Contextual RAG integration failed: {e}")
