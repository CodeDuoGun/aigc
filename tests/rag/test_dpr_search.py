from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的DPR模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = AutoModel.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# 示例文档和查询
documents = ["头痛可能是由多种原因引起的，包括压力、缺乏睡眠或饮食问题。",
             "视力模糊可能是糖尿病的早期症状之一。",
             "高血压可能导致头痛和视力模糊。"]
query = "头痛和视力模糊的原因是什么？"

