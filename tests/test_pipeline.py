#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP管道测试
"""

import os
import tempfile
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.mcp import PipelineComponent, Pipeline


class TestPipelineComponent(PipelineComponent):
    """
    测试用的管道组件
    """
    def __init__(self, name, config=None):
        super().__init__(config)
        self.name = name
        self.processed = False
    
    def process(self, data):
        """
        处理数据
        """
        self.processed = True
        data[f"{self.name}_processed"] = True
        data[f"{self.name}_result"] = f"Result from {self.name}"
        return data


class TestPipeline(unittest.TestCase):
    """
    测试管道功能
    """
    
    def test_pipeline_creation(self):
        """
        测试管道创建
        """
        try:
            # 创建组件
            component1 = TestPipelineComponent("component1")
            component2 = TestPipelineComponent("component2")
            
            # 创建管道
            pipeline = Pipeline([component1, component2])
            self.assertEqual(len(pipeline.components), 2)
            print("✓ 管道创建成功")
        except Exception as e:
            self.fail(f"管道创建失败: {str(e)}")
    
    def test_pipeline_execution(self):
        """
        测试管道执行
        """
        # 创建组件和管道
        component1 = TestPipelineComponent("component1")
        component2 = TestPipelineComponent("component2")
        pipeline = Pipeline([component1, component2])
        
        # 执行管道
        input_data = {"input": "test data"}
        result = pipeline.run(input_data)
        
        # 验证结果
        self.assertTrue(component1.processed)
        self.assertTrue(component2.processed)
        self.assertTrue(result["component1_processed"])
        self.assertTrue(result["component2_processed"])
        self.assertEqual(result["component1_result"], "Result from component1")
        self.assertEqual(result["component2_result"], "Result from component2")
        print("✓ 管道执行成功")
    
    def test_pipeline_component_management(self):
        """
        测试管道组件管理
        """
        # 创建管道
        pipeline = Pipeline()
        
        # 添加组件
        component1 = TestPipelineComponent("component1")
        component2 = TestPipelineComponent("component2")
        pipeline.add_component(component1)
        pipeline.add_component(component2)
        
        # 验证组件添加
        self.assertEqual(len(pipeline.components), 2)
        
        # 清除组件
        pipeline.clear_components()
        self.assertEqual(len(pipeline.components), 0)
        
        print("✓ 管道组件管理成功")


if __name__ == "__main__":
    unittest.main()
