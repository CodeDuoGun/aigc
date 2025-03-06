from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker

texts = [
'哪个快递公司最好？',
'我该选哪家快递？',
'哪个快递最快？',
'哪家快递服务最可靠？',
'我应该用哪个快递寄包裹？',
'哪家快递性价比最高？',
'发货用哪个快递公司比较好？',
'哪个快递公司收费最合理？',
'选择哪个快递更安全？',
'哪个快递公司的客户服务最好？',
    '发顺丰快递'
]

documents = []
for idx, text in enumerate(texts):
    metadata = {"idx": idx}
    doc = Document(page_content=text, metadata=metadata)
    documents.append(doc)

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=512)
texts = text_splitter.split_documents(documents)

local_model_name = 'bert-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=local_model_name)

db = FAISS.from_documents(texts, embeddings)
faiss_index = "vectors_db/hln_tb_faiss_index"
db.save_local(faiss_index)

question = "发什么快递？"
answers = db.similarity_search(question, k=3)
print(answers)


# 构造一个 FlagReranker 实例，设置 use_fp16 为 true 可以加快计算速度
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

new_rerank_pairs = [[question, answer.page_content] for answer in answers]
# 计算多对文本间的相关性评分
scores = reranker.compute_score(new_rerank_pairs)
print(scores)