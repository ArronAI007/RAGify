from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool
import logging
import time


class RAGifyAgent(ABC):
    """
    RAGify Agent 基类
    所有具体Agent实现必须继承此类
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: Agent配置
        """
        self.name = name
        self.config = config or {}
        self.tools = []
        self.logger = logging.getLogger(f"ragify.agents.{self.__class__.__name__}")
        
        # 初始化Agent
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        初始化Agent，子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent调用
        
        Args:
            query: 用户查询
            **kwargs: 额外参数
            
        Returns:
            调用结果
        """
        pass
    
    def add_tool(self, tool: Union[Tool, Dict[str, Any], Callable]) -> None:
        """
        添加工具到Agent
        
        Args:
            tool: Tool对象、工具配置字典或可调用函数
        """
        if isinstance(tool, Tool):
            self.tools.append(tool)
        elif isinstance(tool, dict):
            # 从字典创建Tool (LangChain 1.0 格式)
            name = tool.get("name")
            func = tool.get("func")
            description = tool.get("description", "")
            if name and func:
                self.tools.append(Tool.from_function(name=name, func=func, description=description))
            else:
                self.logger.warning(f"无效的工具配置: {tool}")
        elif callable(tool):
            # 从可调用函数创建Tool (LangChain 1.0 格式)
            name = tool.__name__
            description = tool.__doc__ or ""
            self.tools.append(Tool.from_function(name=name, func=tool, description=description))
    
    def add_tools(self, tools: List[Union[Tool, Dict[str, Any], Callable]]) -> None:
        """
        批量添加工具
        
        Args:
            tools: 工具列表
        """
        for tool in tools:
            self.add_tool(tool)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取Agent信息
        
        Returns:
            Agent信息字典
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "tool_count": len(self.tools),
            "tool_names": [t.name for t in self.tools],
            "config": self.config
        }


class RAGifyTool:
    """
    RAGify 工具基类
    提供工具的标准接口和功能
    """
    
    def __init__(self, name: str, description: str):
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
        """
        self.name = name
        self.description = description
    
    def to_langchain_tool(self, func: Callable) -> Tool:
        """
        转换为LangChain Tool对象
        
        Args:
            func: 工具函数
            
        Returns:
            LangChain Tool对象
        """
        return Tool.from_function(
            name=self.name,
            func=func,
            description=self.description
        )


class AgentRegistry:
    """
    Agent注册表
    管理所有可用的Agent类型
    """
    
    def __init__(self):
        self.agents = {}
    
    def register(self, agent_class: type, name: Optional[str] = None):
        """
        注册Agent类
        
        Args:
            agent_class: Agent类
            name: 注册名称，如果为None则使用类名
        """
        if not issubclass(agent_class, RAGifyAgent):
            raise ValueError(f"类 {agent_class.__name__} 不是 RAGifyAgent 的子类")
        
        reg_name = name or agent_class.__name__
        self.agents[reg_name] = agent_class
        return agent_class
    
    def get_agent_class(self, name: str) -> type:
        """
        获取Agent类
        
        Args:
            name: Agent名称
            
        Returns:
            Agent类
            
        Raises:
            ValueError: 如果Agent不存在
        """
        if name not in self.agents:
            raise ValueError(f"未注册的Agent: {name}")
        return self.agents[name]
    
    def list_agents(self) -> List[str]:
        """
        列出所有可用的Agent
        
        Returns:
            Agent名称列表
        """
        return list(self.agents.keys())
    
    def create_agent(self, name: str, **kwargs) -> RAGifyAgent:
        """
        创建Agent实例
        
        Args:
            name: Agent名称
            **kwargs: Agent初始化参数
            
        Returns:
            Agent实例
        """
        agent_class = self.get_agent_class(name)
        return agent_class(**kwargs)


# 全局Agent注册表
agent_registry = AgentRegistry()


def register_agent(name: Optional[str] = None):
    """
    注册Agent的装饰器
    
    Args:
        name: 注册名称
        
    Returns:
        装饰器函数
    """
    def decorator(agent_class: type):
        return agent_registry.register(agent_class, name)
    return decorator


class AgentExecutionObserver:
    """
    Agent执行观察器
    用于监控和记录Agent的执行过程
    """
    
    def __init__(self):
        self.execution_history = []
        self.start_time = None
        self.end_time = None
    
    def on_execution_start(self, query: str, **kwargs):
        """
        执行开始时调用
        
        Args:
            query: 用户查询
            **kwargs: 额外参数
        """
        self.start_time = time.time()
        self.execution_history.append({
            "type": "execution_start",
            "query": query,
            "timestamp": self.start_time,
            "kwargs": kwargs
        })
    
    def on_tool_call(self, tool_name: str, tool_input: str, tool_output: str):
        """
        工具调用时调用
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入
            tool_output: 工具输出
        """
        self.execution_history.append({
            "type": "tool_call",
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output,
            "timestamp": time.time()
        })
    
    def on_execution_end(self, result: Dict[str, Any]):
        """
        执行结束时调用
        
        Args:
            result: 执行结果
        """
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        self.execution_history.append({
            "type": "execution_end",
            "result": result,
            "timestamp": self.end_time,
            "execution_time": execution_time
        })
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        获取执行摘要
        
        Returns:
            执行摘要信息
        """
        tool_calls = [h for h in self.execution_history if h["type"] == "tool_call"]
        
        return {
            "tool_calls_count": len(tool_calls),
            "tool_calls": [
                {
                    "name": h["tool_name"],
                    "input": h["input"]
                }
                for h in tool_calls
            ],
            "execution_time": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    def reset(self):
        """
        重置观察器状态
        """
        self.execution_history = []
        self.start_time = None
        self.end_time = None
