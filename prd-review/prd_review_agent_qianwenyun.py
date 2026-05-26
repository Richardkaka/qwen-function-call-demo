# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
import docx
from pypdf import PdfReader
from dotenv import load_dotenv

# ====================== 模型导入：千问云(OpenAI兼容) ======================
from langchain_openai import ChatOpenAI

# ===================== 双重API密钥配置（适配千问云） ====================
# 加载同级 .env 文件
load_dotenv()

# 方式1：小白直接在此填写千问云 API Key（优先生效）
API_KEY = ""

# 优先级：代码内密钥 > .env 中的 QIANWEN_API_KEY
QIANWEN_API_KEY = API_KEY.strip() or os.getenv("QIANWEN_API_KEY", "")

# 密钥校验
if not QIANWEN_API_KEY:
    print("❌ 错误：未配置千问云 API Key！")
    print("👉 配置方式：")
    print("   1. 直接在代码 API_KEY = '' 中填写密钥")
    print("   2. 同级 .env 文件中写入：QIANWEN_API_KEY=你的密钥")
    exit(1)

# ===================== 数据模型定义 ====================
class IssueSeverity(str, Enum):
    P0_BLOCKER = "P0-阻断"
    P1_CRITICAL = "P1-严重"
    P2_NORMAL = "P2-一般"
    P3_SUGGESTION = "P3-建议"

class ReviewIssue(BaseModel):
    issue_id: str = Field(description="问题唯一标识")
    severity: IssueSeverity = Field(description="严重程度")
    location: str = Field(description="问题位置")
    description: str = Field(description="问题描述")
    impact: str = Field(description="问题影响")
    suggestion: str = Field(description="修改建议")

class RoleReviewResult(BaseModel):
    role_name: str
    reviewer_name: str
    review_time: str
    overall_opinion: str
    issues: List[ReviewIssue]
    pass_status: bool

class FinalReviewReport(BaseModel):
    prd_name: str
    prd_version: str
    prd_file_path: str
    review_date: str
    selected_roles: List[str]
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_role: Dict[str, int]
    role_results: List[RoleReviewResult]
    final_conclusion: str
    next_steps: List[str]

class RoleRecommendation(BaseModel):
    product_type: str
    core_features: List[str]
    recommended_roles: List[str]
    explanation: str

@dataclass
class ReviewRole:
    name: str
    abbreviation: str
    system_prompt: str
    is_essential: bool = True

# ===================== 评审核心类 ====================
class PRDReviewAgent:
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RoleReviewResult)
        self.json_parser = JsonOutputParser(pydantic_object=RoleRecommendation)
        self.roles = self._init_roles()
        
    def _init_roles(self) -> List[ReviewRole]:
        return [
            ReviewRole(
                name="产品经理",
                abbreviation="PM",
                system_prompt="""你是一位资深产品经理，基于第一性原理+张一鸣产品思维评审需求。
你的评审重点：
1. 【第一性原理】回归需求本质：剥离表面功能，明确需求解决的核心用户/业务问题，是否为伪需求
2. 【第一性原理】需求底层逻辑是否成立，是否基于客观事实而非主观臆想
3. 【张一鸣思维】用户价值第一：是否优先满足用户核心价值，而非单纯追求业务指标
4. 【张一鸣思维】极简高效：流程是否冗余复杂，是否符合极简、信息高效原则
5. 【张一鸣思维】数据驱动：需求是否有明确的数据验证标准
6. 【张一鸣思维】小步快跑：需求是否可拆分迭代，无一次性过度设计
7. 【张一鸣思维】闭环思维：需求从触发到结束是否完整闭环，无缺失环节

注意：你只需要指出现有设计中的问题和漏洞，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="前端开发工程师",
                abbreviation="FE",
                system_prompt="""你是一位资深前端开发工程师，精通交互设计与用户体验实现。
你的评审重点：
1. 交互、页面需求是否有更优的前端实现方案，是否存在体验缺陷
2. 需求是否存在前端技术不可行、兼容性差的问题
3. 响应式、操作流畅度的产品设计是否合理

注意：你只需要指出现有设计中的技术问题和实现难点，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="后端开发工程师",
                abbreviation="BE",
                system_prompt="""你是一位资深后端开发工程师，精通业务逻辑与技术实现拆分。
你的评审重点：
1. 需求是否有更优的后端技术实现方案，当前设计是否存在后端实现缺陷
2. 专项检查：产品标注的【前端实现需求】是否更适合后端实现（如数据计算、安全逻辑、核心业务规则）
3. 专项检查：产品标注的【后端实现需求】是否更适合前端实现（如交互校验、页面渲染、非核心逻辑）
4. 需求的并发、异常、数据边界场景是否清晰，是否存在技术不可行性
5. 第三方依赖、数据流转的需求设计是否合理，有无更优方案

注意：你只需要指出现有设计中的技术问题和实现难点，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="技术架构师",
                abbreviation="ARCH",
                system_prompt="""你是一位资深技术架构师，具备前沿技术视野和系统设计经验。
你的评审重点：
1. 需求对应的实现方案是否存在技术落后、扩展性受限的问题，是否会制约后续业务迭代
2. 是否有更优、更前沿的技术方案能低成本、高效满足需求背后的真实业务目标
3. 当前产品设计是否存在技术架构层面的缺陷（如与现有系统不兼容、单点风险、无法扩容）
4. 方案是否存在过度设计/过于简单，导致无法匹配业务长期发展
5. 跨系统、跨模块的需求设计，是否有更合理的架构整合方案

注意：你只需要指出架构层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="测试工程师",
                abbreviation="QA",
                system_prompt="""你是一位资深测试工程师，精通需求验证、缺陷挖掘与可测试性。
你的评审重点：
1. 需求是否存在前后矛盾、逻辑冲突的问题
2. 需求描述是否模糊、不明确、存在歧义
3. 需求是否存在不合理、不规范、无法落地的设计
4. 需求清晰可测试，无模糊、矛盾逻辑
5. 异常、边界场景是否覆盖，是否存在测试盲区

注意：你只需要指出测试层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="UI设计师",
                abbreviation="UI",
                system_prompt="""你是一位有6年经验的资深UI设计师，精通移动端和Web端设计规范。
你的评审重点：
1. 视觉设计是否符合公司设计规范和品牌调性
2. 色彩、字体、间距、图标等设计元素是否统一
3. 不同状态的视觉表现是否有明确区分
4. 可点击元素的视觉反馈是否清晰
5. 无障碍设计是否有考虑（色盲、色弱用户）
6. 动效设计是否合理，是否会影响性能
7. 切图和标注的交付标准是否明确

注意：你只需要指出现有设计中的问题和漏洞，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="UX设计师",
                abbreviation="UX",
                system_prompt="""你是一位有7年经验的资深UX设计师，专注于用户体验研究。
你的评审重点：
1. 用户流程是否顺畅，是否存在不必要的步骤
2. 用户的认知负担是否过重
3. 信息架构是否合理，用户是否能快速找到所需信息
4. 操作反馈是否及时且明确
5. 新手用户和老用户的需求是否都有考虑
6. 是否存在容易让用户误解的设计
7. 核心功能的使用门槛是否过高

注意：你只需要指出现有设计中的体验问题，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="产品运营",
                abbreviation="OP",
                system_prompt="""你是一位有6年经验的资深产品运营，精通用户增长和活动运营。
你的评审重点：
1. 产品上线后的运营策略是否可行
2. 用户拉新、留存、转化的关键节点是否有埋点
3. 运营后台的功能是否满足日常运营需求
4. 内容审核和管理功能是否完善
5. 用户反馈的收集和处理机制是否明确
6. 版本迭代和灰度发布的支持是否足够
7. 是否存在运营风险（如羊毛党、刷量）

注意：你只需要指出现有设计中与运营相关的问题，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="安全工程师",
                abbreviation="SEC",
                system_prompt="""你是一位资深安全工程师，精通产品安全与权限设计。
你的评审重点：
1. 用户权限、敏感操作、数据脱敏的产品设计是否存在安全漏洞
2. 核心流程（支付、登录、删除）是否有安全风险，易导致用户损失

注意：你只需要指出安全层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="运维工程师",
                abbreviation="OPS",
                system_prompt="""你是一位资深运维工程师，精通系统稳定性与运维成本。
你的评审重点：
1. 需求是否会增加运维成本、难度，是否有更易运维的设计方案
2. 灰度发布、故障排查、扩容的产品设计是否合理

注意：你只需要指出运维层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="数据分析师",
                abbreviation="DA",
                system_prompt="""你是一位有6年经验的资深数据分析师，精通用户行为分析和数据建模。
你的评审重点：
1. 核心指标的定义是否清晰且可计算
2. 埋点方案是否完整，是否能支撑所有数据分析需求
3. 数据上报的时机和频率是否合理
4. 数据口径是否统一，是否存在歧义
5. A/B测试的设计是否科学
6. 数据报表和看板的需求是否明确
7. 是否存在数据隐私和合规问题

注意：你只需要指出数据层面的问题和漏洞，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="客服代表",
                abbreviation="CS",
                system_prompt="""你是一位资深客服专员，熟悉用户痛点、投诉场景与客服成本控制。
你的评审重点：
1. 哪些功能设计极易引发用户抱怨、投诉，需提前优化
2. 哪些功能会大幅增加客服咨询/处理成本，是否需要简化/优化
3. 哪些功能逻辑模糊，用户极易产生疑问，需产品明确说明/补充指引
4. 流程设计是否反人类，是否会导致用户反复找客服协助

注意：你只需要指出可能引发客服问题的设计缺陷，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="大数据工程师",
                abbreviation="BIGDATA",
                system_prompt="""你是一位资深大数据/算法工程师，精通数据处理与智能方案设计。
你的评审重点：
1. 数据处理、算法相关需求是否有更优的实现方案，当前设计是否冗余/低效
2. 是否有前沿大数据/算法技术能更好解决需求，提升效率/用户体验
3. 数据来源、统计口径、算法逻辑的产品设计是否存在缺陷
4. 数据存储、计算、可视化的需求是否有更合理的技术方案

注意：你只需要指出大数据层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="算法工程师",
                abbreviation="ALGO",
                system_prompt="""你是一位资深大数据/算法工程师，精通数据处理与智能方案设计。
你的评审重点：
1. 数据处理、算法相关需求是否有更优的实现方案，当前设计是否冗余/低效
2. 是否有前沿大数据/算法技术能更好解决需求，提升效率/用户体验
3. 数据来源、统计口径、算法逻辑的产品设计是否存在缺陷
4. 数据存储、计算、可视化的需求是否有更合理的技术方案

注意：你只需要指出算法层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="法律合规专员",
                abbreviation="LEGAL",
                system_prompt="""你是一位专业合规专员，精通互联网资质、数据安全、监管合规要求。
你的评审重点：
1. 新增功能是否需要新增资质（如ICP备案、行业许可证、电信资质、数据安全资质等）
2. 现有资质是否覆盖产品功能，是否存在资质不合规风险
3. 用户数据收集、使用、存储是否符合合规要求，有无合规漏洞
4. 功能设计是否违反监管规定，是否需要调整以满足合规要求

注意：你只需要指出法律合规层面的问题和风险，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。"""
            ),
            ReviewRole(
                name="微信小程序审核专员",
                abbreviation="PLATFORM",
                system_prompt="""你是一位有5年经验的资深微信小程序审核专员，精通微信公众平台最新审核规则。
你的评审重点：
1. 产品功能是否符合微信小程序审核规范
2. 用户权限的申请是否合理且必要（特别是地理位置、手机号权限）
3. 支付流程是否符合微信支付规定
4. 内容审核机制是否满足微信要求
5. 隐私政策和用户协议的展示是否符合要求
6. 是否存在诱导分享、诱导下载、诱导关注等违规行为
7. 小程序的类目和描述是否准确，是否存在超范围经营

注意：你只需要指出可能导致微信审核不通过的问题，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            ),
            ReviewRole(
                name="应用商店审核专员",
                abbreviation="APPSTORE",
                system_prompt="""你是一位有5年经验的资深应用商店审核专员，精通苹果App Store和安卓应用市场审核规则。
你的评审重点：
1. 产品功能是否符合应用商店审核规范
2. 用户权限的申请是否合理且必要
3. 支付流程是否符合应用商店规定
4. 内容审核机制是否满足应用商店要求
5. 隐私政策和用户协议的展示是否符合要求
6. 是否存在诱导分享、诱导下载等违规行为
7. 应用的分类和描述是否准确

注意：你只需要指出可能导致应用商店审核不通过的问题，不要提出新的功能需求。
严格按照要求的JSON格式输出评审结果，不要添加任何额外的解释或markdown格式。""",
                is_essential=False
            )
        ]

    def read_prd_file(self, file_path: str) -> str:
        """读取本地PRD文件，支持多格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PRD文件不存在：{file_path}")
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            return '\n'.join([page.extract_text() for page in reader.pages])
        elif ext in ['.xls', '.xlsx']:
            import pandas as pd
            return pd.read_excel(file_path).to_string(index=False)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

    def auto_select_roles(self, prd_content: str) -> RoleRecommendation:
        """AI智能分析PRD内容，自动推荐评审角色"""
        print("正在智能分析PRD内容，自动推荐评审角色...")
        prd_summary = prd_content[:3000] + "..." if len(prd_content) > 3000 else prd_content
        roles_info = "\n".join([f"- {r.abbreviation}: {r.name}" for r in self.roles])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位有15年经验的互联网产品总监，精通各种类型产品的PRD评审流程。
你的任务是分析PRD文档内容，识别产品类型和核心功能，然后推荐最合适的评审角色组合。

所有可用的评审角色：
{roles_info}

推荐原则：
1. 必须包含所有基础角色：PM, FE, BE, QA, UI, UX, DA, CS, LEGAL
2. 根据产品类型添加对应的平台审核角色：
   - 微信小程序：添加PLATFORM
   - iOS/安卓APP：添加APPSTORE
3. 根据产品复杂度添加高级角色：
   - 涉及复杂系统架构：添加ARCH
   - 涉及用户敏感数据或支付：添加SEC
   - 涉及服务器部署和运维：添加OPS
   - 涉及大数据处理：添加BIGDATA
   - 涉及推荐算法、机器学习：添加ALGO
4. 不要添加与产品无关的角色
5. 推荐理由要具体说明为什么需要这些角色"""),
            ("human", """请分析以下PRD文档内容，推荐最合适的评审角色组合：
{prd_summary}
{format_instructions}""")
        ])

        chain = prompt | self.llm | self.json_parser
        result = chain.invoke({
            "roles_info": roles_info,
            "prd_summary": prd_summary,
            "format_instructions": self.json_parser.get_format_instructions()
        })
        return RoleRecommendation(**result)

    def review_by_role(self, role: ReviewRole, prd_content: str, prd_name: str, prd_version: str) -> RoleReviewResult:
        """单个角色评审"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", role.system_prompt),
            ("human", """请对以下PRD文档进行全面评审：
PRD名称：{prd_name}
PRD版本：{prd_version}
PRD内容：{prd_content}
{format_instructions}

重要提醒：
- 只输出纯JSON格式，不要添加任何其他内容
- 不要在JSON前后添加markdown代码块标记
- 不要添加任何解释性文字
- 如果没有发现任何问题，issues字段返回空数组""")
        ])

        chain = prompt | self.llm | self.parser
        return chain.invoke({
            "prd_name": prd_name,
            "prd_version": prd_version,
            "prd_content": prd_content,
            "format_instructions": self.parser.get_format_instructions()
        })

    def full_review(self, prd_file_path: str, prd_name: str, prd_version: str,
                   selected_roles: Optional[List[str]] = None,
                   use_auto_role_selection: bool = True) -> FinalReviewReport:
        """执行全角色PRD评审"""
        print(f"正在读取PRD文件：{prd_file_path}")
        prd_content = self.read_prd_file(prd_file_path)
        print(f"PRD文件读取成功，共{len(prd_content)}个字符")

        # 确定评审角色
        if selected_roles:
            print(f"使用手动指定的评审角色：{', '.join(selected_roles)}")
            roles_to_review = [r for r in self.roles if r.name in selected_roles or r.abbreviation in selected_roles]
        elif use_auto_role_selection:
            role_recommendation = self.auto_select_roles(prd_content)
            print(f"\nAI智能角色推荐结果：")
            print(f"  产品类型：{role_recommendation.product_type}")
            print(f"  核心功能：{', '.join(role_recommendation.core_features)}")
            print(f"  推荐角色：{', '.join(role_recommendation.recommended_roles)}")
            print(f"  推荐理由：{role_recommendation.explanation}")
            roles_to_review = [r for r in self.roles if r.abbreviation in role_recommendation.recommended_roles]
        else:
            print("使用默认的必须评审角色")
            roles_to_review = [r for r in self.roles if r.is_essential]

        print(f"\n开始评审，共{len(roles_to_review)}个角色参与")
        print("-" * 50)

        role_results = []
        for i, role in enumerate(roles_to_review, 1):
            print(f"[{i}/{len(roles_to_review)}] 正在由 {role.name} 进行评审...")
            try:
                res = self.review_by_role(role, prd_content, prd_name, prd_version)
                role_results.append(res)
                print(f"  ✅ 完成，发现{len(res.issues)}个问题")
            except Exception as e:
                print(f"  ❌ 失败：{str(e)}")
                continue

        print("-" * 50)
        print("所有角色评审完成，正在生成报告...")

        # 统计问题
        total_issues = sum(len(r.issues) for r in role_results)
        issues_by_severity = {"P0-阻断": 0, "P1-严重": 0, "P2-一般": 0, "P3-建议": 0}
        issues_by_role = {}

        for res in role_results:
            issues_by_role[res.role_name] = len(res.issues)
            for issue in res.issues:
                issues_by_severity[issue.severity] += 1

        has_p0 = issues_by_severity["P0-阻断"] > 0
        p1_count = issues_by_severity["P1-严重"]

        if has_p0:
            conclusion = "评审不通过。存在P0级阻断性问题，必须全部解决后才能进入开发阶段。"
        elif p1_count > 3:
            conclusion = "评审不通过。存在多个P1级严重问题，建议修改后重新评审。"
        elif p1_count > 0:
            conclusion = "有条件通过。存在少量P1级严重问题，需要在开发前解决并同步给所有相关人员。"
        else:
            conclusion = "评审通过。可以进入开发阶段，P2和P3级问题可在开发过程中逐步解决。"

        next_steps = []
        if has_p0:
            next_steps.append("立即修复所有P0级阻断性问题")
        if p1_count > 0:
            next_steps.append("修复所有P1级严重问题")
        next_steps.append("根据评审意见修改PRD文档")
        next_steps.append("将修改后的PRD同步给所有相关人员")
        if has_p0 or p1_count > 3:
            next_steps.append("组织第二轮专项评审")
        else:
            next_steps.append("进入开发排期阶段")

        return FinalReviewReport(
            prd_name=prd_name,
            prd_version=prd_version,
            prd_file_path=prd_file_path,
            review_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            selected_roles=[r.abbreviation for r in roles_to_review],
            total_issues=total_issues,
            issues_by_severity=issues_by_severity,
            issues_by_role=issues_by_role,
            role_results=role_results,
            final_conclusion=conclusion,
            next_steps=next_steps
        )

    def generate_report_markdown(self, report: FinalReviewReport) -> str:
        """生成Markdown评审报告"""
        md = f"# PRD全维度评审报告\n\n"
        md += f"**PRD名称**：{report.prd_name}  \n"
        md += f"**PRD版本**：{report.prd_version}  \n"
        md += f"**PRD文件**：{report.prd_file_path}  \n"
        md += f"**评审日期**：{report.review_date}  \n"
        md += f"**参与评审角色**：{', '.join(report.selected_roles)}  \n\n"

        md += "## 评审概览\n\n"
        md += f"**总问题数**：{report.total_issues}  \n\n"

        md += "### 问题严重程度分布\n\n"
        md += "| 严重程度 | 问题数量 |\n|----------|----------|\n"
        for k, v in report.issues_by_severity.items():
            md += f"| {k} | {v} |\n"
        md += "\n"

        md += "### 各角色问题统计\n\n"
        md += "| 评审角色 | 问题数量 |\n|----------|----------|\n"
        for k, v in report.issues_by_role.items():
            md += f"| {k} | {v} |\n"
        md += "\n## 最终评审结论\n\n**%s**\n\n" % report.final_conclusion

        md += "## 下一步行动\n\n"
        for idx, step in enumerate(report.next_steps, 1):
            md += f"{idx}. {step}\n"
        md += "\n## 各角色详细评审意见\n\n"

        for res in report.role_results:
            md += f"### {res.role_name}\n\n"
            md += f"**整体意见**：{res.overall_opinion}  \n"
            md += f"**评审状态**：{'✅ 通过' if res.pass_status else '❌ 不通过'}  \n"
            md += f"**发现问题**：{len(res.issues)}个\n\n"
            if res.issues:
                md += "| 问题ID | 严重程度 | 位置 | 问题描述 | 影响 | 建议方案 |\n"
                md += "|--------|----------|------|----------|------|----------|\n"
                for item in res.issues:
                    md += f"| {item.issue_id} | {item.severity} | {item.location} | {item.description} | {item.impact} | {item.suggestion} |\n"
            md += "\n"
        return md

# ===================== 主程序入口 ====================
if __name__ == "__main__":
    # 初始化千问云模型（OpenAI 兼容接口）
    llm = ChatOpenAI(
        model="qwen-turbo",       # 模型：可选 qwen-max / qwen-plus / qwen-turbo
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 千问云固定接口地址
        api_key=QIANWEN_API_KEY,
        temperature=0.1,   # 评审场景保持低随机性
        timeout=120
    )

    review_agent = PRDReviewAgent(llm)

    # 扫描当前目录所有支持的文档
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SUPPORTED_FORMATS = ["*.doc", "*.docx", "*.pdf", "*.xls", "*.xlsx", "*.md", "*.txt"]
    PRD_FILE_LIST = []
    for fmt in SUPPORTED_FORMATS:
        PRD_FILE_LIST.extend(glob.glob(os.path.join(current_dir, fmt)))

    if not PRD_FILE_LIST:
        print("❌ 当前目录未找到任何支持的文档！")
        exit(1)

    print(f"✅ 找到 {len(PRD_FILE_LIST)} 个待评审文档：")
    for i, file in enumerate(PRD_FILE_LIST, 1):
        print(f" {i}. {os.path.basename(file)}")

    # 批量评审所有文档
    for idx, file_path in enumerate(PRD_FILE_LIST, 1):
        file_name = os.path.basename(file_path)
        print(f"\n===== 开始评审第 {idx} 个文档：{file_name} =====")

        report = review_agent.full_review(
            prd_file_path=file_path,
            prd_name=file_name,
            prd_version="V1.0",
            use_auto_role_selection=True
        )

        # 保存报告
        md_content = review_agent.generate_report_markdown(report)
        report_name = f"PRD评审报告_{file_name}_{datetime.now().strftime('%Y%m%d%H%M')}.md"
        with open(report_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"✅ 文档 {file_name} 评审完成！报告已保存为：{report_name}")
        print(f"共发现 {report.total_issues} 个问题")
        print(f"最终结论：{report.final_conclusion}")