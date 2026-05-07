#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 管道测试
验证 Pipeline 和 PipelineComponent 的核心行为。
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.mcp.base import PipelineComponent, Pipeline


class AddOneComponent(PipelineComponent):
    """测试组件：将 data['value'] 加 1"""

    def run(self, data):
        data["value"] = data.get("value", 0) + 1
        return data


class DoubleComponent(PipelineComponent):
    """测试组件：将 data['value'] 翻倍"""

    def run(self, data):
        data["value"] = data.get("value", 0) * 2
        return data


class TestPipelineComponent(unittest.TestCase):
    """测试 PipelineComponent 基类"""

    def test_execute_runs_run(self):
        comp = AddOneComponent("add_one")
        result = comp.execute({"value": 5})
        self.assertEqual(result["value"], 6)

    def test_disabled_component_skipped(self):
        comp = AddOneComponent("add_one", config={"enabled": False})
        result = comp.execute({"value": 5})
        self.assertEqual(result["value"], 5)

    def test_preprocess_and_postprocess(self):
        class PrePostComponent(PipelineComponent):
            def preprocess(self, data):
                data["pre"] = True
                return data

            def run(self, data):
                data["run"] = True
                return data

            def postprocess(self, data):
                data["post"] = True
                return data

        comp = PrePostComponent("pre_post")
        result = comp.execute({})
        self.assertTrue(result["pre"])
        self.assertTrue(result["run"])
        self.assertTrue(result["post"])

    def test_get_info(self):
        comp = AddOneComponent("add_one")
        info = comp.get_info()
        self.assertEqual(info["name"], "add_one")
        self.assertEqual(info["type"], "AddOneComponent")
        self.assertTrue(info["enabled"])


class TestPipeline(unittest.TestCase):
    """测试 Pipeline 类"""

    def test_pipeline_creation(self):
        pipeline = Pipeline("test_pipeline")
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertEqual(len(pipeline.components), 0)

    def test_add_and_remove_component(self):
        pipeline = Pipeline("test_pipeline")
        comp = AddOneComponent("add_one")
        pipeline.add_component(comp)
        self.assertEqual(len(pipeline.components), 1)

        removed = pipeline.remove_component("add_one")
        self.assertTrue(removed)
        self.assertEqual(len(pipeline.components), 0)

    def test_get_component(self):
        pipeline = Pipeline("test_pipeline")
        comp = AddOneComponent("add_one")
        pipeline.add_component(comp)
        found = pipeline.get_component("add_one")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "add_one")

    def test_pipeline_execution_order(self):
        pipeline = Pipeline("math_pipeline")
        pipeline.add_component(AddOneComponent("add_one"))  # 5 + 1 = 6
        pipeline.add_component(DoubleComponent("double"))   # 6 * 2 = 12
        result = pipeline.run({"value": 5})
        self.assertEqual(result["value"], 12)

    def test_disabled_pipeline_returns_input(self):
        pipeline = Pipeline("disabled", config={"enabled": False})
        pipeline.add_component(AddOneComponent("add_one"))
        result = pipeline.run({"value": 5})
        self.assertEqual(result["value"], 5)

    def test_pipeline_validation(self):
        pipeline = Pipeline("test")
        pipeline.add_component(AddOneComponent("add_one"))
        pipeline.add_component(DoubleComponent("double"))
        self.assertTrue(pipeline.validate())

    def test_pipeline_validation_duplicate_names(self):
        pipeline = Pipeline("test")
        pipeline.add_component(AddOneComponent("same_name"))
        pipeline.add_component(DoubleComponent("same_name"))
        self.assertFalse(pipeline.validate())


if __name__ == "__main__":
    unittest.main()
