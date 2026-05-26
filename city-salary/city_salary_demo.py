# -*- coding: utf-8 -*-
import json
import os
import dashscope
from dotenv import load_dotenv

# ===================== 🔥 新增：双重API Key配置（唯一修改区） =====================
# 加载.env文件
load_dotenv()
# 小白直接填这里，进阶用户用.env
API_KEY = ""

# 双重加载：代码优先 → 其次.env
dashscope.api_key = API_KEY.strip() or os.getenv("DASHSCOPE_API_KEY")

# 无API Key直接退出
if not dashscope.api_key:
    print("❌ 请配置API Key：代码内填写 或 同级目录放.env文件")
    exit(1)
# ==============================================================================

# ====================== 1. 预设全国主流城市人均月薪模拟数据（25座城市） ======================
# 仅作学习演示模拟数据，非官方真实统计数据
CITY_SALARY_DATA = {
    "北京": 9800,
    "上海": 10200,
    "广州": 7500,
    "深圳": 8900,
    "杭州": 8200,
    "南京": 7100,
    "成都": 6200,
    "重庆": 5900,
    "武汉": 6500,
    "西安": 5800,
    "长沙": 6000,
    "郑州": 5700,
    "青岛": 6300,
    "济南": 6100,
    "天津": 6800,
    "苏州": 7900,
    "无锡": 7300,
    "宁波": 7600,
    "佛山": 6400,
    "东莞": 6600,
    "合肥": 5600,
    "福州": 6000,
    "昆明": 5300,
    "沈阳": 5500,
    "大连": 5400
}

# ====================== 2. 通用工具函数：查询城市人均薪资 ======================
def get_city_average_salary(target_city: str, salary_type: str = "社会通用薪资") -> str:
    """
    获取指定城市平均薪资模拟数据
    :param target_city: 目标查询城市名称
    :param salary_type: 薪资分类类型
    :return: 结构化JSON格式薪资信息
    """
    avg_salary = CITY_SALARY_DATA.get(target_city, None)
    if not avg_salary:
        result_info = {
            "city": target_city,
            "salary": "暂无收录该城市薪资数据",
            "salary_type": salary_type,
            "income_level_analysis": "暂无相关收入水平分析"
        }
    else:
        # 通用地域收入水平分析
        if avg_salary >= 8000:
            analysis = "该城市收入水平偏高，整体消费层级较高"
        elif 6000 <= avg_salary < 8000:
            analysis = "该城市收入水平中等，大众生活压力适中"
        else:
            analysis = "该城市收入水平偏低，日常生活成本相对友好"
            
        result_info = {
            "city": target_city,
            "average_salary": f"{avg_salary} 元/月",
            "salary_type": salary_type,
            "income_level_analysis": analysis
        }
    # 保留中文正常显示
    return json.dumps(result_info, ensure_ascii=False)

# ====================== 3. 大模型函数调用描述配置 ======================
SALARY_QUERY_FUNCTION = [
    {
        "name": "get_city_average_salary",
        "description": "查询国内主流城市人均月度收入薪资数据，用于了解不同城市大众收入水平",
        "parameters": {
            "type": "object",
            "properties": {
                "target_city": {
                    "type": "string",
                    "description": "需要查询收入薪资的城市名称"
                },
                "salary_type": {
                    "type": "string",
                    "enum": ["社会通用薪资", "服务行业薪资", "普通行业薪资"],
                    "description": "选择对应行业薪资类型，无指定默认通用薪资"
                }
            },
            "required": ["target_city"]
        }
    }
]

# ====================== 4. 封装通义千问模型统一调用接口 ======================
def call_qwen_model(chat_messages):
    """调用阿里通义千问大模型，统一处理请求与异常"""
    try:
        model_response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=chat_messages,
            functions=SALARY_QUERY_FUNCTION,
            result_format="message"
        )
        return model_response
    except Exception as error:
        print(f"大模型接口调用异常：{str(error)}")
        return None

# ====================== 5. 核心对话流程：智能意图识别+函数调用 ======================
def city_salary_chat(user_question: str):
    """
    智能对话入口
    自动识别用户意图，判断是否需要调用薪资查询工具
    """
    chat_records = [{"role": "user", "content": user_question}]
    
    # 首次请求模型：意图识别
    first_response = call_qwen_model(chat_records)
    if not first_response or not first_response.output:
        return "请求数据失败，请检查网络与API密钥配置"
    
    ai_first_reply = first_response.output.choices[0].message
    chat_records.append(ai_first_reply)

    # ========== 修复点：改用字典方式判断function_call，避免KeyError ==========
    # 正确判断模型是否要调用工具（不触发KeyError）
    if "function_call" in ai_first_reply and ai_first_reply["function_call"]:
        call_info = ai_first_reply["function_call"]
        func_name = call_info["name"]
        call_params = json.loads(call_info["arguments"])
        
        # 执行本地自定义函数
        city_name = call_params.get("target_city")
        job_salary_type = call_params.get("salary_type", "社会通用薪资")
        local_data_result = get_city_average_salary(city_name, job_salary_type)
        
        # 封装函数执行结果，回传大模型
        func_result_msg = {
            "role": "function",
            "name": func_name,
            "content": local_data_result
        }
        chat_records.append(func_result_msg)
        
        # 二次调用模型：整合数据生成自然语言回答
        final_response = call_qwen_model(chat_records)
        if not final_response or not final_response.output:
            return "数据整合解析失败"
        
        final_answer = final_response.output.choices[0].message.content
        return final_answer
    
    # ========== 修复点：处理无关问题的回复 ==========
    # 无需调用工具，直接返回模型原生对话回复（确保content不为空）
    if ai_first_reply.get("content"):
        return ai_first_reply["content"]
    else:
        # 兜底回复，避免空字符串
        return "抱歉，我只能帮你查询国内主流城市的人均薪资数据哦~"

# ====================== 程序入口 测试运行 ======================
if __name__ == "__main__":
    # 通用测试提问（无任何业务相关内容）
    test_questions = [
        "成都大众平均月收入大概多少",
        "深圳普通行业薪资水平怎么样",
        "今天天气怎么样",
        "郴州服务行业人均薪资是多少"
    ]
    
    for question in test_questions:
        print(f"\n【用户提问】：{question}")
        res = city_salary_chat(question)
        print(f"【AI智能回复】：{res}")
        print("-" * 70)