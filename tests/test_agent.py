#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 模块测试
验证 Agent 基类、注册表和工具行为。
"""

import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.agents.base import (
    RAGifyAgent, RAGifyTool, AgentRegistry, agent_registry, register_agent, AgentExecutionObserver
)
from langchain_core.tools import Tool


class DummyAgent(RAGifyAgent):
    """测试用的最小 Agent 实现"""

    def __init__(self, name: str = "dummy", config=None):
        self.name = name
        self.config = config or {}
        self.tools = []
        self.logger = logging.getLogger("test.DummyAgent")
        self._initialize()

    def _initialize(self) -> None:
        pass

    def invoke(self, query: str, **kwargs):
        return {"query": query, "response": f"echo: {query}"}


class TestAgentRegistry(unittest.TestCase):
    """测试 AgentRegistry"""

    def test_register_and_create(self):
        registry = AgentRegistry()
        registry.register(DummyAgent, "dummy")
        self.assertIn("dummy", registry.list_agents())

        # create_agent 的 agent_type 参数是注册表中的 agent 名称
        # kwargs 会传递给 Agent 的构造函数
        instance = registry.create_agent("dummy", name="test_dummy")
        self.assertIsInstance(instance, DummyAgent)
        self.assertEqual(instance.name, "test_dummy")

    def test_register_non_agent_raises(self):
        registry = AgentRegistry()
        with self.assertRaises(ValueError):
            registry.register(str, "string")

    def test_get_missing_agent_raises(self):
        registry = AgentRegistry()
        with self.assertRaises(ValueError):
            registry.get_agent_class("missing")

    def test_global_registry_has_builtin_agents(self):
        # 全局注册表在导入时已注册内置 Agent
        from ragify.agents.rag_agent import RAGAgent, MultiModalRAGAgent, PipelineAgent
        self.assertIn("RAGAgent", agent_registry.list_agents())
        self.assertIn("MultiModalRAGAgent", agent_registry.list_agents())
        self.assertIn("PipelineAgent", agent_registry.list_agents())

    def test_register_agent_decorator(self):
        @register_agent("decorated_agent")
        class DecoratedAgent(RAGifyAgent):
            def _initialize(self):
                pass

            def invoke(self, query, **kwargs):
                return {"response": "decorated"}

        self.assertIn("decorated_agent", agent_registry.list_agents())


class TestRAGifyAgent(unittest.TestCase):
    """测试 RAGifyAgent 基类"""

    def test_init_requires_name(self):
        agent = DummyAgent(name="my_agent")
        self.assertEqual(agent.name, "my_agent")
        self.assertEqual(agent.tools, [])

    def test_add_tool_from_dict(self):
        agent = DummyAgent(name="test")
        agent.add_tool({
            "name": "dummy_tool",
            "func": lambda x: x,
            "description": "A dummy tool",
        })
        self.assertEqual(len(agent.tools), 1)
        self.assertIsInstance(agent.tools[0], Tool)

    def test_add_tool_from_callable(self):
        def my_func(x):
            return x

        agent = DummyAgent(name="test")
        agent.add_tool(my_func)
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(agent.tools[0].name, "my_func")

    def test_add_invalid_tool_ignored(self):
        agent = DummyAgent(name="test")
        agent.add_tool({"name": "bad", "func": None})
        self.assertEqual(len(agent.tools), 0)

    def test_get_info(self):
        agent = DummyAgent(name="info_test")
        info = agent.get_info()
        self.assertEqual(info["name"], "info_test")
        self.assertEqual(info["type"], "DummyAgent")


class TestRAGifyTool(unittest.TestCase):
    """测试 RAGifyTool"""

    def test_to_langchain_tool(self):
        tool = RAGifyTool(name="my_tool", description="Does something")
        lc_tool = tool.to_langchain_tool(lambda x: f"result: {x}")
        self.assertIsInstance(lc_tool, Tool)
        self.assertEqual(lc_tool.name, "my_tool")


class TestAgentExecutionObserver(unittest.TestCase):
    """测试 AgentExecutionObserver"""

    def test_execution_summary(self):
        obs = AgentExecutionObserver()
        obs.on_execution_start("query")
        obs.on_tool_call("search", "input", "output")
        obs.on_execution_end({"output": "result"})

        summary = obs.get_execution_summary()
        self.assertEqual(summary["tool_calls_count"], 1)
        self.assertEqual(summary["tool_calls"][0]["name"], "search")
        self.assertGreaterEqual(summary["execution_time"], 0)

    def test_reset(self):
        obs = AgentExecutionObserver()
        obs.on_execution_start("query")
        obs.reset()
        summary = obs.get_execution_summary()
        self.assertEqual(summary["execution_time"], 0)


if __name__ == "__main__":
    unittest.main()
