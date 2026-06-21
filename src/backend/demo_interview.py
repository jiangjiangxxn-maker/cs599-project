"""
面试模拟演示脚本
模拟完整的面试流程：开场 → 多轮问答 → 最终评估
"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.core.state import OrchestratorState, AgentContext
from app.core.memory import memory_manager
from app.orchestrator.graph import orchestrator
from app.agents.agent_4_interviewer import InterviewerAgent


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"{'='*70}")


def print_message(role, content):
    """打印对话消息"""
    if role == "user":
        print(f"\n👤 候选人: {content}")
    else:
        print(f"\n🤖 面试官: {content}")


def print_qa_item(qa, turn_num):
    """打印 Q&A 项"""
    print(f"\n📝 第 {turn_num} 轮:")
    print(f"  Q: {qa.question}")
    if qa.user_answer:
        print(f"  A: {qa.user_answer[:100]}...")
    if qa.score:
        print(f"  ⭐ 评分: {qa.score}/10")
    if qa.feedback:
        print(f"  💬 反馈: {qa.feedback[:100]}...")


async def demo_interview():
    """演示完整的面试流程"""
    
    print_separator("🎤 面试模拟演示 - 后端技术面试")
    print("\n场景：应届生后端开发岗位技术面试")
    print("面试官：AI 面试官")
    print("候选人：模拟用户")
    
    # 初始化内存系统
    await memory_manager.initialize()
    
    # 禁用跨 Agent 自动唤醒（避免面试结束时触发学习规划器导致卡住）
    original_post_process = orchestrator._post_process
    async def mock_post_process(state):
        """跳过跨 Agent 唤醒，直接返回状态"""
        return state
    orchestrator._post_process = mock_post_process
    
    # 创建会话
    session_id = memory_manager.create_session()
    state = memory_manager.get_session(session_id)
    state.agent_context = AgentContext(current_agent_id=-1)
    
    # ========================================================================
    # 预设的面试流程
    # ========================================================================
    
    # 用户输入序列（与面试官问题相匹配的回答）
    user_inputs = [
        "模拟后端技术面试",  # 初始请求
        "整个过程包括：1. DNS解析将域名转换为IP地址 2. 建立TCP连接（三次握手） 3. 发送HTTP/HTTPS请求 4. 服务器处理请求并返回响应 5. 浏览器解析HTML/CSS/JS并渲染页面",  # Q1 回答：URL访问过程
        "HTTP是无状态的应用层协议，默认端口80，数据明文传输；HTTPS在HTTP基础上加入SSL/TLS加密，默认端口443，数据加密传输，更安全但性能略低",  # Q2 回答：HTTP vs HTTPS
        "TCP三次握手：1. 客户端发送SYN包到服务器 2. 服务器返回SYN+ACK包 3. 客户端发送ACK包。目的是确保双方都有发送和接收能力，同步序列号",  # Q3 回答：TCP三次握手
        "synchronized是Java关键字，自动释放锁；ReentrantLock是API级别，需要手动释放锁，支持公平锁、可中断、可超时等更灵活的特性",  # Q4 回答：synchronized vs ReentrantLock
        "结束面试",  # 结束请求
    ]
    
    # 面试官问题（预设，用于展示）
    expected_questions = [
        "请解释 Java 中 HashMap 的实现原理",
        "ConcurrentHashMap 如何保证线程安全？",
        "Redis 和 Memcached 有什么区别？",
    ]
    
    print_separator("开始面试")
    
    # 逐轮进行面试
    for turn_idx, user_input in enumerate(user_inputs, 1):
        print_separator(f"第 {turn_idx} 轮")
        
        # 设置用户输入
        state.user_input = user_input
        
        # 保存用户消息到历史
        state.agent_context.conversation_history.append({
            "role": "user",
            "content": user_input,
            "agent_id": -1,
            "agent": "User",
        })
        
        # 打印用户输入
        print_message("user", user_input)
        
        # 通过 orchestrator 处理
        print("\n⏳ 面试官正在思考...")
        state = await orchestrator.process(state)
        
        # 获取最新的 assistant 消息
        assistant_msgs = [msg for msg in state.agent_context.conversation_history 
                         if msg.get("role") == "assistant" and msg.get("agent_id") == 4]
        
        if assistant_msgs:
            last_msg = assistant_msgs[-1]
            print_message("assistant", last_msg["content"])
            
            # 如果是面试会话，显示详细信息
            if state.agent_context.interview_session:
                session = state.agent_context.interview_session
                if session.qa_history:
                    latest_qa = session.qa_history[-1]
                    print_qa_item(latest_qa, turn_idx)
        
        # 检查面试是否结束
        if not state.agent_context.is_interviewing and turn_idx > 1:
            print("\n✅ 面试已结束")
            break
        
        # 模拟延迟
        await asyncio.sleep(0.5)
    
    # ========================================================================
    # 显示最终评估报告
    # ========================================================================
    print_separator("📊 最终评估报告")
    
    if state.agent_context.interview_session:
        session = state.agent_context.interview_session
        
        print(f"\n🎯 面试模式: {session.mode}")
        print(f"📝 总轮数: {len(session.qa_history)}")
        print(f"\n⭐ 总分: {session.overall_score:.1f}/10")
        
        if session.strengths:
            print(f"\n✅ 优势:")
            for i, strength in enumerate(session.strengths, 1):
                print(f"  {i}. {strength}")
        
        if session.weaknesses:
            print(f"\n⚠️  薄弱点:")
            for i, weakness in enumerate(session.weaknesses, 1):
                print(f"  {i}. {weakness}")
        
        if session.improvement_plan:
            print(f"\n💡 改进建议:")
            for i, plan in enumerate(session.improvement_plan, 1):
                print(f"  {i}. {plan}")
    
    # ========================================================================
    # 显示完整对话历史
    # ========================================================================
    print_separator("📚 完整对话历史")
    
    for msg in state.agent_context.conversation_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        agent = msg.get("agent", "")
        
        if role == "user":
            print(f"\n👤 候选人: {content[:100]}...")
        elif role == "assistant" and agent == "Interviewer":
            print(f"\n🤖 面试官: {content[:100]}...")
    
    # ========================================================================
    # 验证修复：检查对话历史中的 assistant 消息
    # ========================================================================
    print_separator("🔍 验证对话历史修复")
    
    assistant_msgs = [msg for msg in state.agent_context.conversation_history 
                     if msg.get("role") == "assistant"]
    
    print(f"\n找到 {len(assistant_msgs)} 条 assistant 消息")
    
    type_strings = ["career_feasibility_report", "jd_analysis_report", "resume_optimization", 
                    "interview_simulation", "learning_plan", "job_matching", "processed"]
    
    all_valid = True
    for idx, msg in enumerate(assistant_msgs, 1):
        content = msg.get("content", "")
        is_type_string = content in type_strings
        
        status = "❌ 错误" if is_type_string else "✅ 正确"
        print(f"\n{idx}. {status} - 长度: {len(content)} 字符")
        
        if is_type_string:
            print(f"   内容: {content}")
            all_valid = False
    
    if all_valid:
        print("\n✅ 所有 assistant 消息都包含格式化内容，修复成功！")
    else:
        print("\n❌ 发现问题，需要进一步修复")
    
    print_separator("演示完成")
    print("\n💡 提示：")
    print("  - 在实际前端中，用户输入'模拟后端技术面试'即可开始")
    print("  - 面试官会逐轮提问并评分")
    print("  - 输入'结束面试'或达到最大轮数后自动结束")
    print("  - 最终会生成详细的评估报告")


if __name__ == "__main__":
    asyncio.run(demo_interview())
