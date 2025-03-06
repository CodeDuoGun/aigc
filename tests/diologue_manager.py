import re

# 1. 预处理
def preprocess(text):
    return re.sub(r'[^\w\s]', '', text).lower()

# 2. 语义理解（提取地点）
def extract_location(text):
    match = re.search(r'in\s+(\w+)', text)
    return match.group(1) if match else None

# 3. 对话管理
class WeatherBot:
    def __init__(self):
        self.location = None
    
    def respond(self, text):
        cleaned_text = preprocess(text)
        location = extract_location(cleaned_text) or self.location
        if not location:
            return "Please tell me a location."
        self.location = location
        return f"Weather in {location}: Sunny, 25°C."

# 4. 生成回复
bot = WeatherBot()
print(bot.respond("What's the weather like?"))        # 输出：请提供地点
print(bot.respond("What about in Berlin?"))           # 输出：Berlin的天气：晴，25°C