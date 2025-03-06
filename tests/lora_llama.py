"""
pip install transformers peft accelerate datasets bitsandbytes
# 可选：安装支持Flash Attention的库（提升训练速度）
pip install flash-attn --no-build-isolation

"""

"""
标准格式：JSON/Lines 文件，每行包含 {"text": "..."}

"""
import torch
# 
from datasets import load_dataset
dataset = load_dataset("json", data_files="data/lora.json")

from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型与Tokenizer加载
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4-bit量化节省显存
    device_map="auto",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

# 配置LoRA参数
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,            # 低秩矩阵的秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标模块（LLaMA3的注意力层）
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数（应远小于原始模型）


# 训练配置
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,  # 混合精度训练
    save_steps=500,
    optim="paged_adamw_8bit"  # 8-bit优化器
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=lambda data: {
        "input_ids": torch.stack([f["input_ids"] for f in data]),
        "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        "labels": torch.stack([f["input_ids"] for f in data])
    }
)
trainer.train()
model.save_pretrained("llama3-lora")  # 保存适配器权重

# 推理测试
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "llama3-lora")

inputs = tokenizer("请解释人工智能:", return_tensors="pt")
outputs = model.generate(inputs=inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))