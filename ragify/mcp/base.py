from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from langchain_core.documents import Document


class PipelineComponent(ABC):
    """
    Pipeline组件的抽象基类，定义了所有组件必须实现的接口
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self._setup()
    
    def _setup(self) -> None:
        """
        组件初始化设置，可以在子类中重写
        """
        pass
    
    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行组件的核心逻辑
        
        Args:
            data: 输入数据字典
            
        Returns:
            处理后的输出数据字典
        """
        pass
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理输入数据
        
        Args:
            data: 原始输入数据
            
        Returns:
            预处理后的数据
        """
        return data
    
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理输出数据
        
        Args:
            data: 处理后的原始数据
            
        Returns:
            后处理后的数据
        """
        return data
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行组件的完整处理流程
        
        Args:
            data: 输入数据
            
        Returns:
            输出数据
        """
        if not self.enabled:
            print(f"组件 {self.name} 已禁用，跳过执行")
            return data
        
        try:
            # 预处理
            preprocessed_data = self.preprocess(data)
            
            # 核心处理
            processed_data = self.run(preprocessed_data)
            
            # 后处理
            final_data = self.postprocess(processed_data)
            
            return final_data
        except Exception as e:
            print(f"执行组件 {self.name} 时出错: {e}")
            # 出错时返回原始数据
            return data
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取组件信息
        
        Returns:
            组件信息字典
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "enabled": self.enabled
        }


class Pipeline:
    """
    处理管道，由多个PipelineComponent组成
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.components: List[PipelineComponent] = []
        self.enabled = self.config.get("enabled", True)
    
    def add_component(self, component: PipelineComponent) -> None:
        """
        添加组件到管道
        
        Args:
            component: 要添加的组件
        """
        self.components.append(component)
    
    def add_components(self, components: List[PipelineComponent]) -> None:
        """
        批量添加组件到管道
        
        Args:
            components: 组件列表
        """
        self.components.extend(components)
    
    def remove_component(self, component_name: str) -> bool:
        """
        从管道中移除组件
        
        Args:
            component_name: 组件名称
            
        Returns:
            是否成功移除
        """
        for i, component in enumerate(self.components):
            if component.name == component_name:
                self.components.pop(i)
                return True
        return False
    
    def get_component(self, component_name: str) -> Optional[PipelineComponent]:
        """
        获取管道中的组件
        
        Args:
            component_name: 组件名称
            
        Returns:
            组件实例，如果不存在返回None
        """
        for component in self.components:
            if component.name == component_name:
                return component
        return None
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行整个管道
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理后的输出数据
        """
        if not self.enabled:
            print(f"管道 {self.name} 已禁用，跳过执行")
            return input_data
        
        data = input_data.copy()
        
        # 记录管道执行信息
        execution_info = {
            "pipeline_name": self.name,
            "components_executed": [],
            "start_data_size": len(str(data))
        }
        data["_pipeline_info"] = execution_info
        
        # 依次执行每个组件
        for component in self.components:
            # 检查组件是否启用
            if component.enabled:
                print(f"执行组件: {component.name}")
                data = component.execute(data)
                execution_info["components_executed"].append(component.name)
            else:
                print(f"跳过禁用的组件: {component.name}")
        
        # 更新执行信息
        execution_info["end_data_size"] = len(str(data))
        execution_info["success"] = True
        
        return data
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取管道信息
        
        Returns:
            管道信息字典
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "component_count": len(self.components),
            "components": [comp.get_info() for comp in self.components]
        }
    
    def validate(self) -> bool:
        """
        验证管道配置是否正确
        
        Returns:
            是否验证通过
        """
        try:
            # 检查组件名称是否重复
            component_names = set()
            for component in self.components:
                if component.name in component_names:
                    print(f"错误: 组件名称重复: {component.name}")
                    return False
                component_names.add(component.name)
            
            # 检查必要的组件是否存在
            # 可以在子类中添加更多验证逻辑
            
            return True
        except Exception as e:
            print(f"验证管道时出错: {e}")
            return False
