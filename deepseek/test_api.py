# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import requests
import json

TOKEN = "sk-be8a57b06be64700baf1534d1e60963a"
client = OpenAI(api_key=TOKEN, base_url="https://api.deepseek.com")
print(client.models.list())
response = client.chat.completions.create(
    model="deepseek-chat",# deepseek v3
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)


def fim():
    url = "https://api.deepseek.com/beta/completions"

    payload = json.dumps({
    "model": "deepseek-chat",
    "prompt": "Once upon a time, ",
    "echo": False,
    "frequency_penalty": 0,
    "logprobs": 0,
    "max_tokens": 1024,
    "presence_penalty": 0,
    "stop": None,
    "stream": False,
    "stream_options": None,
    "suffix": None, # 制定被补全内容的后缀
    "temperature": 1,
    "top_p": 1
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {TOKEN}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

def banlance():
    """check user balance"""
    url = "https://api.deepseek.com/user/balance"

    payload={}
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {TOKEN}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    print(response.text)

def deepseek_reason():
    # Round 1
    messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    messages.append(response.choices[0].message)
    print(f"Messages Round 1: {messages}")

    # Round 2
    messages.append({"role": "user", "content": "What is the second?"})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    messages.append(response.choices[0].message)
    print(f"Messages Round 2: {messages}")


def create_prompt():
    """构建一个提示词生成任务"""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.deepseek.com/",
        api_key="sk-be8a57b06be64700baf1534d1e60963a"
    )

    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                    "role": "system",
                    "content": "你是一位大模型提示词生成专家，请根据用户的需求编写一个智能助手的提示词，来指导大模型进行内容生成，要求：\n1. 以 Markdown 格式输出\n2. 贴合用户需求，描述智能助手的定位、能力、知识储备\n3. 提示词应清晰、精确、易于理解，在保持质量的同时，尽可能简洁\n4. 只输出提示词，不要输出多余解释"
            },
            {
                    "role": "user",
                    "content": "请帮我生成一个“Linux 助手”的提示词"
            }
        ]
    )

    print(completion.choices[0].message.content)

def main():
    # deepseek_reason()
    create_prompt()
    
if __name__ == "__main__":
    main()
    # fim()
    # banlance()