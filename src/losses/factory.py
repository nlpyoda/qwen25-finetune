"""
Loss function factory for creating and managing custom losses.

This module provides a centralized way to register, configure,
and instantiate different loss functions from YAML configurations.
"""

from typing import Dict, Type, Any, Optional, Union, List
import yaml
from pathlib import Path
import importlib

from .base import BaseLoss, MultiComponentLoss, LossOutput
from .custom_losses import (
    FocalLoss, LabelSmoothingLoss, ContrastiveLoss, 
    KLDivergenceLoss, CurriculumLoss, MultimodalAlignmentLoss,
    TokenTypeLoss
)

class LossFactory:
    """
    Factory class for creating and managing loss functions.
    
    This factory allows easy registration and instantiation of custom
    loss functions, enabling seamless switching between different
    loss configurations through YAML files.
    
    Example:
        ```python
        # Register a custom loss
        factory = LossFactory()
        factory.register_loss("MyCustomLoss", MyCustomLoss)
        
        # Create loss from config
        loss_fn = factory.create_loss("my_loss_config.yaml")
        
        # Use in training
        loss_output = loss_fn(logits, labels)
        ```
    """
    
    def __init__(self):
        self._losses: Dict[str, Type[BaseLoss]] = {}
        
        # Register built-in losses
        self._register_builtin_losses()
    
    def _register_builtin_losses(self):
        """Register built-in loss functions."""
        builtin_losses = {
            "focal": FocalLoss,
            "label_smoothing": LabelSmoothingLoss,
            "contrastive": ContrastiveLoss,
            "kl_divergence": KLDivergenceLoss,
            "curriculum": CurriculumLoss,
            "multimodal_alignment": MultimodalAlignmentLoss,
            "token_type": TokenTypeLoss,
            "multi_component": MultiComponentLoss
        }
        
        for name, loss_class in builtin_losses.items():
            self.register_loss(name, loss_class)
    
    def register_loss(self, name: str, loss_class: Type[BaseLoss]):
        """
        Register a loss function.
        
        Args:
            name: Name to register the loss under
            loss_class: The loss class to register
        """
        if not issubclass(loss_class, BaseLoss):
            raise ValueError(f"Loss class {loss_class} must inherit from BaseLoss")
        
        self._losses[name] = loss_class
        print(f"Registered loss function: {name}")
    
    def register_from_module(self, module_path: str, name: str):
        """
        Register a loss function from a module path.
        
        Args:
            module_path: Python module path (e.g., "my_package.my_loss")
            name: Name to register under
        """
        try:
            module = importlib.import_module(module_path)
            
            # Look for loss class
            loss_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseLoss) and 
                    attr != BaseLoss):
                    loss_class = attr
                    break
            
            if loss_class:
                self.register_loss(name, loss_class)
            else:
                raise ValueError(f"No valid loss class found in {module_path}")
                
        except ImportError as e:
            raise ImportError(f"Could not import module {module_path}: {e}")
    
    def create_loss(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        loss_name: Optional[str] = None,
        **kwargs
    ) -> BaseLoss:
        """
        Create a loss function from configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
            loss_name: Direct loss name for registered losses
            **kwargs: Additional arguments to pass to loss constructor
            
        Returns:
            Instantiated loss function
        """
        if config_path:
            config = self._load_config_from_file(config_path)
        elif config_dict:
            config = config_dict
        elif loss_name:
            config = {"loss": {"type": loss_name}}
        else:
            raise ValueError("Must provide config_path, config_dict, or loss_name")
        
        return self._create_from_config(config, **kwargs)
    
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_from_config(self, config: Dict[str, Any], **kwargs) -> BaseLoss:
        """Create loss function from configuration dictionary."""
        loss_config = config.get("loss", {})
        
        if isinstance(loss_config, list):
            # Multi-component loss configuration
            return self._create_multi_component_loss(loss_config, **kwargs)
        else:
            # Single loss configuration
            return self._create_single_loss(loss_config, **kwargs)
    
    def _create_single_loss(self, loss_config: Dict[str, Any], **kwargs) -> BaseLoss:
        """Create a single loss function."""
        loss_type = loss_config.get("type")
        
        if not loss_type:
            raise ValueError("Loss configuration must specify 'type'")
        
        if loss_type not in self._losses:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                           f"Available: {list(self._losses.keys())}")
        
        loss_class = self._losses[loss_type]
        
        # Extract loss-specific configuration
        loss_params = loss_config.get("config", {})
        loss_params.update(kwargs)
        
        # Handle special parameters
        if "weight" in loss_config:
            loss_params["weight"] = loss_config["weight"]
        
        return loss_class(**loss_params)
    
    def _create_multi_component_loss(
        self, 
        loss_configs: List[Dict[str, Any]], 
        **kwargs
    ) -> MultiComponentLoss:
        """Create a multi-component loss function."""
        loss_components = {}
        
        for i, loss_config in enumerate(loss_configs):
            component_name = loss_config.get("name", f"component_{i}")
            component_loss = self._create_single_loss(loss_config, **kwargs)
            loss_components[component_name] = component_loss
        
        # Check for multi-component specific configuration
        multi_config = kwargs.get("multi_component_config", {})
        normalize_weights = multi_config.get("normalize_weights", False)
        
        return MultiComponentLoss(
            loss_components=loss_components,
            normalize_weights=normalize_weights,
            **{k: v for k, v in kwargs.items() if k != "multi_component_config"}
        )
    
    def list_losses(self) -> Dict[str, str]:
        """List all registered loss functions."""
        return {
            name: loss_class.__name__ for name, loss_class in self._losses.items()
        }
    
    def get_loss_info(self, loss_name: str) -> Dict[str, Any]:
        """Get information about a specific loss function."""
        if loss_name not in self._losses:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        loss_class = self._losses[loss_name]
        
        # Extract docstring and parameters
        info = {
            "name": loss_name,
            "class_name": loss_class.__name__,
            "description": loss_class.__doc__.strip() if loss_class.__doc__ else "No description",
            "module": loss_class.__module__
        }
        
        # Try to extract parameter information from __init__
        import inspect
        try:
            signature = inspect.signature(loss_class.__init__)
            parameters = {}
            for param_name, param in signature.parameters.items():
                if param_name != "self":
                    param_info = {
                        "type": str(param.annotation) if param.annotation != param.empty else "Any",
                        "default": param.default if param.default != param.empty else "Required"
                    }
                    parameters[param_name] = param_info
            info["parameters"] = parameters
        except Exception:
            info["parameters"] = "Could not extract parameter information"
        
        return info
    
    def save_loss_registry(self, output_path: str):
        """Save information about all registered losses."""
        registry_info = {
            "registered_losses": {},
            "builtin_count": len([name for name in self._losses.keys() 
                                 if not name.startswith("custom_")]),
            "custom_count": len([name for name in self._losses.keys() 
                               if name.startswith("custom_")])
        }
        
        for loss_name in self._losses.keys():
            try:
                registry_info["registered_losses"][loss_name] = self.get_loss_info(loss_name)
            except Exception as e:
                registry_info["registered_losses"][loss_name] = {
                    "error": f"Could not get info: {e}"
                }
        
        with open(output_path, 'w') as f:
            yaml.dump(registry_info, f, default_flow_style=False)
    
    def create_loss_from_examples(self) -> Dict[str, BaseLoss]:
        """Create example loss functions for demonstration."""
        examples = {}
        
        # Focal loss example
        examples["focal_loss"] = self.create_loss(config_dict={
            "loss": {
                "type": "focal",
                "config": {"alpha": 1.0, "gamma": 2.0, "weight": 1.0}
            }
        })
        
        # Label smoothing example
        examples["label_smoothing"] = self.create_loss(config_dict={
            "loss": {
                "type": "label_smoothing",
                "config": {"smoothing": 0.1, "weight": 1.0}
            }
        })
        
        # Multi-component loss example
        examples["multi_component"] = self.create_loss(config_dict={
            "loss": [
                {
                    "name": "main_loss",
                    "type": "label_smoothing",
                    "config": {"smoothing": 0.1},
                    "weight": 1.0
                },
                {
                    "name": "auxiliary_loss",
                    "type": "focal",
                    "config": {"gamma": 2.0},
                    "weight": 0.1
                }
            ]
        })
        
        return examples

# Global loss factory instance
loss_factory = LossFactory()

def register_loss(name: str, loss_class: Type[BaseLoss]):
    """Convenience function to register a loss."""
    loss_factory.register_loss(name, loss_class)

def create_loss(config_path: str, **kwargs) -> BaseLoss:
    """Convenience function to create a loss."""
    return loss_factory.create_loss(config_path, **kwargs)

def list_available_losses() -> Dict[str, str]:
    """Convenience function to list available losses."""
    return loss_factory.list_losses()

class LossScheduler:
    """
    Scheduler for dynamically adjusting loss weights during training.
    
    This class allows implementing sophisticated loss scheduling strategies
    such as gradually increasing auxiliary loss weights or implementing
    curriculum learning schedules.
    """
    
    def __init__(
        self,
        loss_function: BaseLoss,
        schedule_config: Dict[str, Any]
    ):
        """
        Initialize loss scheduler.
        
        Args:
            loss_function: The loss function to schedule
            schedule_config: Configuration for scheduling
        """
        self.loss_function = loss_function
        self.schedule_config = schedule_config
        self.current_step = 0
        
        # Parse schedule configuration
        self.schedule_type = schedule_config.get("type", "constant")
        self.milestones = schedule_config.get("milestones", [])
        self.factors = schedule_config.get("factors", [])
        self.warmup_steps = schedule_config.get("warmup_steps", 0)
    
    def step(self) -> Dict[str, float]:
        """
        Perform a scheduling step.
        
        Returns:
            Dictionary of current weights/parameters
        """
        self.current_step += 1
        
        if self.schedule_type == "milestone":
            self._apply_milestone_schedule()
        elif self.schedule_type == "linear_warmup":
            self._apply_linear_warmup()
        elif self.schedule_type == "cosine_warmup":
            self._apply_cosine_warmup()
        
        return self.get_current_weights()
    
    def _apply_milestone_schedule(self):
        """Apply milestone-based weight scheduling."""
        for i, milestone in enumerate(self.milestones):
            if self.current_step >= milestone and i < len(self.factors):
                if hasattr(self.loss_function, 'weight'):
                    self.loss_function.weight *= self.factors[i]
    
    def _apply_linear_warmup(self):
        """Apply linear warmup scheduling."""
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            if hasattr(self.loss_function, 'weight'):
                base_weight = self.schedule_config.get("base_weight", 1.0)
                self.loss_function.weight = base_weight * warmup_factor
    
    def _apply_cosine_warmup(self):
        """Apply cosine warmup scheduling."""
        if self.current_step <= self.warmup_steps:
            import math
            warmup_factor = 0.5 * (1 + math.cos(math.pi * self.current_step / self.warmup_steps))
            if hasattr(self.loss_function, 'weight'):
                base_weight = self.schedule_config.get("base_weight", 1.0)
                self.loss_function.weight = base_weight * warmup_factor
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        weights = {"step": self.current_step}
        
        if hasattr(self.loss_function, 'weight'):
            weights["main_weight"] = self.loss_function.weight
        
        # For multi-component losses
        if isinstance(self.loss_function, MultiComponentLoss):
            for name, component in self.loss_function.loss_components.items():
                weights[f"{name}_weight"] = component.weight
        
        return weights