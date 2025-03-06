import re
from nltk.tokenize import word_tokenize

def text_preprocessing(text):
    # 去除非字母数字字符
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词（示例列表）
    stop_words = {"is", "the", "a"}
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

input_text = "Hello! How's it going??"
print(text_preprocessing(input_text))  # 输出：['hello', 'how', 's', 'it', 'going']