# 使用transformers库加载LLaMA模型和Tokenizer。例如，加载LLaMA-3-8B模型：
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name)
# 配置PEFT方法
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=16,  # LoRA矩阵的秩
    lora_alpha=32, # LoRA的alpha参数
    lora_dropout=0.1, # 10%的dropout
    bias="none", # 不使用bias
    task_type="CAUSAL_LM" # 
)
model = get_peft_model(model, peft_config)


# 使用datasets库加载微调数据集。例如，加载一个简单的问答数据集：
from datasets import load_dataset
dataset = load_dataset("json", data_files="path/to/your/dataset.json", split="train")

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    save_steps=500,
    logging_steps=100,
    fp16=True,
)
# 使用trl库中的SFTTrainer进行监督微调：
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model("./output")
# 微调完成后，保存模型和适配器
model.save_pretrained("./output")

# 如果需要进一步优化模型的推理速度和存储，可以使用AWQ（Adaptive Weight Quantization）对微调后的模型进行量化
# from autoawq import AutoAWQForCausalLM
# quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
# model = AutoAWQForCausalLM.from_pretrained("./output", safetensors=True)
# model.quantize(tokenizer, quant_config=quant_config)
# model.save_quantized("./quantized_output", safetensors=True)
# 以下是一个bert的意图识别case
from transformers import pipeline

# 加载预训练模型
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

text = "I'm really excited about the new product launch!"
result = classifier(text)
print(result)  # 输出：[{'label': 'POSITIVE', 'score': 0.999}]