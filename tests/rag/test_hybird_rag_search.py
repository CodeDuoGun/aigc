from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from rank_bm25 import BM25Okapi
from typing import List

# 1. 加载文档
loader = TextLoader("data/wiki_demo.txt")  # 替换为你的文档路径
documents = loader.load()

# 2. 文本分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. 初始化 BM25 检索器
tokenized_docs = [doc.page_content.split(" ") for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

class CustomBM25Retriever(BM25Retriever):
    def __init__(self, documents):
        super().__init__(documents)
        self.bm25 = bm25

    def get_relevant_documents(self, query, k=3):
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_n = self.bm25.get_top_n(tokenized_query, [doc.page_content for doc in self.documents], n=k)
        return [doc for doc in self.documents if doc.page_content in top_n]

# 4. 初始化向量检索器
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. 创建混合检索器
ensemble_retriever = EnsembleRetriever(retrievers=[CustomBM25Retriever(docs), vector_retriever], weights=[0.5, 0.5])

# 6. 检索示例
query = "What is the capital of France?"
retrieved_docs = ensemble_retriever.get_relevant_documents(query, k=5)

# 7. 输出结果
print("Retrieved Documents:")
for idx, doc in enumerate(retrieved_docs):
    print(f"{idx + 1}. {doc.page_content}")