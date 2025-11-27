#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGify - 多模态RAG系统，支持MCP和Agent功能
"""

from setuptools import setup, find_packages

with open("pyproject.toml", "r", encoding="utf-8") as f:
    import re
    content = f.read()
    
    # 从pyproject.toml中提取必要的信息
    name = re.search(r'name = "([^"]+)"', content).group(1)
    description = re.search(r'description = "([^"]+)"', content).group(1)
    version = re.search(r'version = "([^"]+)"', content).group(1)

setup(
    name=name,
    version=version,
    description=description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain==1.0.5",
        "pydantic>=2.7.0,<3.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pillow>=10.0.0",
        "langchain-community>=0.0.20",
        "langchain-core>=1.0.0",
        "langchain-openai==0.1.0",
        "langchain-anthropic==0.1.0",
    ],
    extras_require={
        "llms": [
            "openai>=1.0.0,<2.0.0",
        ],
        "embeddings": [
            "sentence-transformers>=3.0.0,<4.0.0",
        ],
        "vectorstores": [
            "faiss-cpu>=1.8.0,<2.0.0",
        ],
        "dev": [
            "pytest",
            "black",
            "isort",
            "click",
            "tqdm",
            "ipython"
        ],
    },
    entry_points={
        "console_scripts": [
            "ragify=ragify.cli:ragify",
        ],
    },
    python_requires=">=3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
