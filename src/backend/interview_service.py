"""
独立的面试模拟服务
基于 demo_interview.py 的逻辑，提供 Web API 接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import sys
sys.path.insert(0, '.')

from app.core.state import OrchestratorState, AgentContext
from app.core.memory import memory_manager
from app.orchestrator.graph import orchestrator


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Interview Demo Service",
    description="独立的面试模拟服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class StartInterviewRequest(BaseModel):
    mode: str = "technical"  # technical, hr, project_deep_dive
    position: str = "backend"


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class InterviewSession:
    """面试会话状态"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = None
        self.is_interviewing = False
        self.current_question = None


# 全局会话存储
interview_sessions = {}


# ============================================================================
# Helper Functions
# ============================================================================

def disable_cross_agent_wake():
    """禁用跨 Agent 唤醒，避免面试结束时触发学习规划器"""
    async def mock_post_process(state):
        return state
    orchestrator._post_process = mock_post_process


async def initialize_session(session_id: str) -> OrchestratorState:
    """初始化面试会话"""
    await memory_manager.initialize()
    
    state = memory_manager.get_session(session_id)
    if not state:
        session_id = memory_manager.create_session()
        state = memory_manager.get_session(session_id)
    
    state.agent_context = AgentContext(current_agent_id=-1)
    disable_cross_agent_wake()
    
    return state


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/interview/start")
async def start_interview(request: StartInterviewRequest):
    """
    开始面试
    返回第一个问题和会话ID
    """
    try:
        # 创建新会话
        session_id = memory_manager.create_session()
        state = memory_manager.get_session(session_id)
        state.agent_context = AgentContext(current_agent_id=-1)
        
        # 禁用跨 Agent 唤醒
        disable_cross_agent_wake()
        
        # 构建面试请求
        mode_text = {
            "technical": "模拟后端技术面试",
            "hr": "模拟HR面试",
            "project_deep_dive": "模拟项目深挖面试"
        }.get(request.mode, "模拟后端技术面试")
        
        # 保存用户消息
        state.user_input = mode_text
        state.agent_context.conversation_history.append({
            "role": "user",
            "content": mode_text,
            "agent_id": -1,
            "agent": "User",
        })
        
        # 处理面试请求
        state = await orchestrator.process(state)
        
        # 获取面试官的第一条消息
        assistant_msgs = [msg for msg in state.agent_context.conversation_history 
                         if msg.get("role") == "assistant" and msg.get("agent_id") == 4]
        
        if not assistant_msgs:
            raise HTTPException(status_code=500, detail="面试启动失败")
        
        last_msg = assistant_msgs[-1]
        
        # 保存会话
        interview_sessions[session_id] = InterviewSession(session_id)
        interview_sessions[session_id].state = state
        interview_sessions[session_id].is_interviewing = True
        
        # 提取问题
        question = ""
        if state.agent_context.interview_session and state.agent_context.interview_session.qa_history:
            question = state.agent_context.interview_session.qa_history[-1].question
        
        return {
            "session_id": session_id,
            "message": last_msg["content"],
            "question": question,
            "is_interviewing": state.agent_context.is_interviewing
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")


@app.post("/interview/answer")
async def submit_answer(request: AnswerRequest):
    """
    提交回答
    返回反馈和下一个问题
    """
    try:
        # 获取会话
        if request.session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        session = interview_sessions[request.session_id]
        state = session.state
        
        if not session.is_interviewing:
            return {
                "message": "面试已结束",
                "is_interviewing": False,
                "final_report": get_final_report(state)
            }
        
        # 保存用户回答
        state.user_input = request.answer
        state.agent_context.conversation_history.append({
            "role": "user",
            "content": request.answer,
            "agent_id": -1,
            "agent": "User",
        })
        
        # 处理回答
        state = await orchestrator.process(state)
        session.state = state
        
        # 获取面试官回复
        assistant_msgs = [msg for msg in state.agent_context.conversation_history 
                         if msg.get("role") == "assistant" and msg.get("agent_id") == 4]
        
        if not assistant_msgs:
            raise HTTPException(status_code=500, detail="处理回答失败")
        
        last_msg = assistant_msgs[-1]
        
        # 提取问题和评分
        question = ""
        score = None
        feedback = ""
        
        if state.agent_context.interview_session and state.agent_context.interview_session.qa_history:
            qa = state.agent_context.interview_session.qa_history[-1]
            question = qa.question
            score = qa.score
            feedback = qa.feedback or ""
        
        # 检查面试是否结束
        is_interviewing = state.agent_context.is_interviewing
        
        response = {
            "message": last_msg["content"],
            "question": question,
            "score": score,
            "feedback": feedback,
            "is_interviewing": is_interviewing
        }
        
        # 如果面试结束，添加最终报告
        if not is_interviewing:
            response["final_report"] = get_final_report(state)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理回答失败: {str(e)}")


@app.post("/interview/end")
async def end_interview(session_id: str):
    """
    强制结束面试
    """
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        session = interview_sessions[session_id]
        state = session.state
        
        # 发送结束消息
        state.user_input = "结束面试"
        state.agent_context.conversation_history.append({
            "role": "user",
            "content": "结束面试",
            "agent_id": -1,
            "agent": "User",
        })
        
        # 处理
        state = await orchestrator.process(state)
        session.state = state
        session.is_interviewing = False
        
        return {
            "message": "面试已结束",
            "final_report": get_final_report(state)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结束面试失败: {str(e)}")


@app.get("/interview/status/{session_id}")
async def get_status(session_id: str):
    """
    获取面试状态
    """
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = interview_sessions[session_id]
    state = session.state
    
    return {
        "session_id": session_id,
        "is_interviewing": session.is_interviewing,
        "current_agent": state.agent_context.current_agent_id if state else None
    }


def get_final_report(state: OrchestratorState) -> dict:
    """提取最终评估报告"""
    if not state or not state.agent_context.interview_session:
        return {}
    
    session = state.agent_context.interview_session
    
    return {
        "mode": session.mode,
        "total_turns": len(session.qa_history),
        "overall_score": session.overall_score,
        "strengths": session.strengths,
        "weaknesses": session.weaknesses,
        "improvement_plan": session.improvement_plan
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    args = parser.parse_args()
    
    print(f"🎤 面试模拟服务启动在端口 {args.port}")
    print(f"📝 API 文档: http://localhost:{args.port}/docs")
    
    uvicorn.run(
        "interview_service:app",
        host="0.0.0.0",
        port=args.port,
        reload=True
    )
