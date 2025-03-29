from llama_index.llms.ollama import Ollama


def test_ollama_llm():
    llm = Ollama(model="llama3.1:8b")
    assert llm is not None
    assert llm.model == "llama3.1:8b"
