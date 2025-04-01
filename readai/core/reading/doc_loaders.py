from langchain_community.document_loaders import UnstructuredEPubLoader
from llama_index.core.schema import Document


# 基于Unstructured的EPub加载器
class UnstructuredEPubLoader(UnstructuredEPubLoader):
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)

    def load(self) -> list[Document]:
        return super().load()
