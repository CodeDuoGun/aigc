
## TF-IDF（词频-逆向文件频率）

## 向量数据库
    FAISS
    Milvus
## 向量相似度，向量夹角

## 


# 大模型理论知识
1、LLM 都有哪些，
    1.1 GPT
    1.2 llama
    1.3 国内 ，豆包、千问、GLM

    1.4 区别和差异，局限，优缺点？
        基座大模型（预训练模型），不太能专业回答，所以需要微调适应下游任务
        

    1.5 LLM 的输入，图、文本，文档（split 段落）
## 智谱GLM


2、LLM 回复不准确？ 提示工程

3、LLM 缺乏相关知识？ RAG，让 LLM 更加聪明
    3.1、 向量数据库，存每个token的向量值
    3.2、 Top N，相似度最高的前几个，交给prompt
    3.3、 最后给到LLM，给出答案

4、LLM 能力不足？ 微调

5、私有化部署，自研方向了。。。

## 微调
1、全参微调
2、微参微调 Lora，节约内存，减少训练时间


## Langchain 
    一个开发LLM 驱动的应用程序框架，
    组成部分
    1、components，提供接口封装、模板提示和信息检索
    2、chains，不同组件结合来解决特定任务
    3、agents，llm与外界环境进行交互
## LangSmith 监控工具，需要api key

## 如何管理大量历史数据 trim


## Langserve 部署
    fastapi && langserve.reoterunnable

## Agent 整合数据库


## RAG 
文档（html、json、pdf、csv等） - 切割（文本切割器、base标点，base标题） - embedding - 向量数据库(Chroma) - 检索(retriever) - 和LLM 结合
## langchain 读取数据库

# 论文介绍
## Transformer

## BERT

## GPT
## 智谱的论文
## Lora 方法
## 