import os
import yaml
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class ConfigLoader:
    """
    配置加载器，负责从YAML文件加载配置，并与环境变量集成
    """
    
    def __init__(self, config_path: str = "./config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._resolve_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _resolve_env_vars(self) -> None:
        """
        解析配置中的环境变量引用
        """
        self._resolve_env_vars_recursive(self.config)
    
    def _resolve_env_vars_recursive(self, obj: Any) -> None:
        """
        递归解析环境变量
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("$"):
                    env_var = value[1:]
                    if env_var in os.environ:
                        obj[key] = os.environ[env_var]
                    elif "api_key_env" in key and env_var in os.environ:
                        obj[key] = os.environ[env_var]
                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                self._resolve_env_vars_recursive(item)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持嵌套键
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        """
        return self.config
    
    def update(self, key: str, value: Any) -> None:
        """
        更新配置项
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class RAGifySettings(BaseSettings):
    """
    RAGify系统的Pydantic设置
    """
    # 基本设置
    project_name: str = Field(default="RAGify")
    version: str = Field(default="0.1.0")
    log_level: str = Field(default="INFO")
    data_dir: str = Field(default="./data")
    output_dir: str = Field(default="./output")
    
    # 配置文件路径
    config_path: str = Field(default="./config/config.yaml")
    
    @validator('data_dir', 'output_dir')
    def ensure_absolute_path(cls, v: str) -> str:
        """确保路径为绝对路径"""
        if not os.path.isabs(v):
            return os.path.abspath(v)
        return v
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# 创建全局配置实例
config_loader: Optional[ConfigLoader] = None
settings: Optional[RAGifySettings] = None


def initialize_config(config_path: str = "./config/config.yaml") -> None:
    """
    初始化配置
    """
    global config_loader, settings
    config_loader = ConfigLoader(config_path)
    settings = RAGifySettings(config_path=config_path)
    
    # 创建必要的目录
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)


def get_config() -> ConfigLoader:
    """
    获取配置加载器实例
    """
    if config_loader is None:
        initialize_config()
    return config_loader


def get_settings() -> RAGifySettings:
    """
    获取Pydantic设置实例
    """
    if settings is None:
        initialize_config()
    return settings
