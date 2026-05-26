# -*- coding: utf-8 -*-
"""
通义千问VL 图片内容提取工具
GitHub: https://github.com/Richardkaka/qwen-function-call-demo
"""
import os
import dashscope
from dotenv import load_dotenv

# 加载同级目录的.env文件（标准规范）
load_dotenv()

# ===================== 配置区域 =====================
# 小白直接填这里，进阶用户用.env
API_KEY = ""
# 自动创建同级images文件夹（标准规范）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
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

def process_single_image(image_path):
    try:
        content = [{"image": image_path},{"text": "提取图片内容，输出JSON格式"}]
        messages = [{"role": "user", "content": content}]
        response = get_qwen_response(messages)
        return response.output.choices[0].message.content[0]['text']
    except Exception as e:
        return f"处理失败：{str(e)}"

def batch_process_images():
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"✅ 创建文件夹：{IMAGE_FOLDER}")
        return
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(SUPPORTED_FORMATS)]
    if not image_files:
        print("📂 请将图片放入images文件夹")
        return
    for i, filename in enumerate(image_files, 1):
        full_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"🖼️ 第{i}张：{filename}")
        print(process_single_image(full_path))

if __name__ == "__main__":
    print("🚀 工具启动")
    batch_process_images()