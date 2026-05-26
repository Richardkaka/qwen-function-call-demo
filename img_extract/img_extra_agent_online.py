# -*- coding: utf-8 -*-
"""
通义千问VL 图片内容提取工具（网络图片模式）
GitHub: https://github.com/Richardkaka/qwen-function-call-demo
"""
import os
import dashscope
from dotenv import load_dotenv

# 加载同级目录的.env文件
load_dotenv()

# ===================== 配置区域 =====================
# 小白直接填这里，进阶用户用.env
API_KEY = ""
# 🔥 固定填写你要识别的网络图片地址（直接用你给的链接）
IMAGE_URL = "https://www.shuomingshu.cn/wp-content/uploads/2022/09/1131001191-0_dyvyfkmcrya.jpg"
# 模型配置
MODEL_NAME = 'qwen-vl-plus'
# ====================================================

# 双重加载：代码优先 → 其次.env
dashscope.api_key = API_KEY.strip() or os.getenv("DASHSCOPE_API_KEY")

# 无API Key直接退出
if not dashscope.api_key:
    print("❌ 请配置API Key：代码内填写 或 同级目录放.env文件")
    exit(1)

def get_qwen_response(messages):
    return dashscope.MultiModalConversation.call(model=MODEL_NAME, messages=messages)

# 核心识别代码（完全没动）
def process_image(image_url):
    try:
        content = [
            {"image": image_url},
            {"text": "提取图片内容，输出JSON格式"}
        ]
        messages = [{"role": "user", "content": content}]
        response = get_qwen_response(messages)
        return response.output.choices[0].message.content[0]['text']
    except Exception as e:
        return f"处理失败：{str(e)}"

# 🔥 主函数：直接读取指定的网络图片
if __name__ == "__main__":
    print("🚀 工具启动（指定网络图片模式）")
    print(f"🖼️ 正在识别图片：{IMAGE_URL}")
    result = process_image(IMAGE_URL)
    print("-" * 50)
    print("📝 识别结果：\n", result)