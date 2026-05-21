# AI PRD 智能全维度评审工具
✨ 基于AI大模型的自动化PRD产品需求文档评审工具，无需人工判断参与角色，一键生成专业评审报告

## 核心功能
- 📄 **多格式支持**：支持 PDF / DOCX / MD / TXT 格式PRD文档
- 🤖 **AI智能角色推荐**：自动分析PRD内容，智能匹配需要参与评审的角色
- 👥 **全角色评审**：产品、前端、后端、测试、UI/UX、安全、合规、平台审核等18种专业角色
- 📊 **自动生成报告**：输出结构化Markdown评审报告，问题分级、统计一目了然
- ⚡ **开箱即用**：配置API密钥即可运行，无需复杂环境

## 支持的评审角色
基础角色（必选）：产品经理(PM)、前端(FE)、后端(BE)、测试(QA)、UI/UX、数据分析师(DA)、客服(CS)、法律合规(LEGAL)
智能可选角色：架构师、安全工程师、运维、大数据、算法、微信小程序审核、应用商店审核等

## 快速开始
### 1. 环境依赖安装
```bash   ”“bash
pip3 install langchain langchain-community pydantic python-docx pypdf dashscope
```

### 2. 配置AI大模型密钥
打开 `prd_review_agent.py`，填写你的**阿里云千问API密钥**：
```python
os.environ["DASHSCOPE_API_KEY"] = "你的API密钥"
```

### 3. 修改PRD文档信息
```python
# 仅需修改这3个参数
PRD_FILE_PATH = "./你的PRD文档.pdf"
PRD_NAME = "你的项目名称"
PRD_VERSION = "V1.0"
```

### 4. 运行工具
```bash   ”“bash
python3 prd_review_agent.py
```

### 5. 查看结果
自动生成 `PRD评审报告_xxx.md`，包含所有问题、结论、整改建议

## 技术栈
- Python 3
- LangChain
- 通义千问大模型
- PyPDF / python-docx

## 使用说明
1. 支持本地PDF/Word/文本PRD文档
2. AI自动判断评审角色，无需人工选择
3. 问题分为P0(阻断)/P1(严重)/P2(一般)/P3(建议)四个等级
4. 报告支持直接用于团队评审、开发验收

## 许可证
MIT License   与条款
```
