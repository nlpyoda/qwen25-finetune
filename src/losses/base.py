"""
Base classes for custom loss functions.

This module provides the foundation for creating custom loss functions
that can be easily integrated into the training framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LossOutput:
    """
    Output structure for loss functions.
    
    This standardized output allows loss functions to return multiple
    components and metrics for monitoring and optimization.
    """
    loss: torch.Tensor
    # Additional loss components
    auxiliary_losses: Optional[Dict[str, torch.Tensor]] = None
    # Metrics for monitoring
    metrics: Optional[Dict[str, float]] = None
    # Debugging information
    debug_info: Optional[Dict[str, Any]] = None

class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    
    This class provides a standard interface for implementing custom
    loss functions with support for:
    - Multiple loss components
    - Auxiliary losses (regularization, etc.)
    - Metrics computation
    - Loss weighting and scheduling
    - Debugging and monitoring
    
    To create a custom loss:
    1. Inherit from this class
    2. Implement the compute_loss method
    3. Register it in the loss factory
    
    Example:
        ```python
        class MyCustomLoss(BaseLoss):
            def __init__(self, weight=1.0, temperature=1.0):
                super().__init__(weight=weight)
                self.temperature = temperature
                
            def compute_loss(self, logits, labels, **kwargs):
                # Your custom loss computation
                loss = F.cross_entropy(logits / self.temperature, labels)
                return LossOutput(loss=loss, metrics={"temperature": self.temperature})
        ```
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        """
        Initialize base loss function.
        
        Args:
            weight: Weight factor for this loss component
            reduction: Reduction method ("mean", "sum", "none")
            ignore_index: Index to ignore in loss computation
            label_smoothing: Label smoothing factor
            **kwargs: Additional loss-specific arguments
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        # Store additional configuration
        self.config = kwargs
        
        # Loss scheduling support
        self.current_step = 0
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.schedule_type = kwargs.get("schedule_type", "constant")
        
        # Metrics tracking
        self.loss_history = []
        self.metrics_history = []
    
    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """
        Compute the loss.
        
        This method should be implemented by subclasses to define
        the specific loss computation logic.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            **kwargs: Additional arguments (attention_mask, position_ids, etc.)
            
        Returns:
            LossOutput containing the computed loss and any additional information
        """
        pass
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """
        Forward pass that applies weighting and scheduling.
        
        This method wraps the compute_loss method with additional
        functionality like loss weighting, scheduling, and monitoring.
        """
        # Compute the base loss
        loss_output = self.compute_loss(logits, labels, **kwargs)
        
        # Apply weight scheduling
        current_weight = self._get_current_weight()
        loss_output.loss = loss_output.loss * current_weight
        
        # Apply auxiliary loss weights if present
        if loss_output.auxiliary_losses:
            for aux_name, aux_loss in loss_output.auxiliary_losses.items():
                aux_weight = self.config.get(f"{aux_name}_weight", 1.0)
                loss_output.auxiliary_losses[aux_name] = aux_loss * aux_weight
        
        # Update metrics with weight information
        if loss_output.metrics is None:
            loss_output.metrics = {}
        loss_output.metrics.update({
            "loss_weight": current_weight,
            "step": self.current_step
        })
        
        # Track loss history
        self.loss_history.append(loss_output.loss.item())
        if loss_output.metrics:
            self.metrics_history.append(loss_output.metrics.copy())
        
        # Update step counter
        self.current_step += 1
        
        return loss_output
    
    def _get_current_weight(self) -> float:
        """Get the current loss weight based on scheduling."""
        if self.schedule_type == "constant":
            return self.weight
        elif self.schedule_type == "linear_warmup":
            if self.current_step < self.warmup_steps:
                return self.weight * (self.current_step / self.warmup_steps)
            return self.weight
        elif self.schedule_type == "cosine_warmup":
            if self.current_step < self.warmup_steps:
                warmup_ratio = self.current_step / self.warmup_steps
                return self.weight * (1 - torch.cos(warmup_ratio * torch.pi / 2))
            return self.weight
        else:
            return self.weight
    
    def reset_history(self):
        """Reset loss and metrics history."""
        self.loss_history = []
        self.metrics_history = []
        self.current_step = 0
    
    def get_average_loss(self, last_n: Optional[int] = None) -> float:
        """Get average loss over last n steps."""
        if not self.loss_history:
            return 0.0
        
        history = self.loss_history[-last_n:] if last_n else self.loss_history
        return sum(history) / len(history)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics
        summary = {}
        for metrics in self.metrics_history:
            for key, value in metrics.items():
                if key not in summary:
                    summary[key] = []
                summary[key].append(value)
        
        # Compute statistics
        stats = {}
        for key, values in summary.items():
            if isinstance(values[0], (int, float)):
                stats[f"{key}_mean"] = sum(values) / len(values)
                stats[f"{key}_std"] = torch.std(torch.tensor(values)).item()
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
        
        return stats

class MultiComponentLoss(BaseLoss):
    """
    Loss function that combines multiple loss components.
    
    This class allows combining different loss functions with
    different weights and scheduling strategies.
    
    Example:
        ```python
        loss_fn = MultiComponentLoss({
            "main_loss": CrossEntropyLoss(weight=1.0),
            "auxiliary_loss": KLDivergenceLoss(weight=0.1),
            "regularization": L2RegularizationLoss(weight=0.01)
        })
        ```
    """
    
    def __init__(
        self,
        loss_components: Dict[str, BaseLoss],
        normalize_weights: bool = False,
        **kwargs
    ):
        """
        Initialize multi-component loss.
        
        Args:
            loss_components: Dictionary of named loss components
            normalize_weights: Whether to normalize component weights to sum to 1
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.loss_components = nn.ModuleDict(loss_components)
        self.normalize_weights = normalize_weights
        
        if normalize_weights:
            self._normalize_component_weights()
    
    def _normalize_component_weights(self):
        """Normalize component weights to sum to 1."""
        total_weight = sum(component.weight for component in self.loss_components.values())
        if total_weight > 0:
            for component in self.loss_components.values():
                component.weight = component.weight / total_weight
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Compute combined loss from all components."""
        total_loss = 0.0
        auxiliary_losses = {}
        metrics = {}
        debug_info = {}
        
        for name, component in self.loss_components.items():
            component_output = component(logits, labels, **kwargs)
            
            # Add to total loss
            total_loss = total_loss + component_output.loss
            
            # Collect auxiliary losses
            if component_output.auxiliary_losses:
                for aux_name, aux_loss in component_output.auxiliary_losses.items():
                    auxiliary_losses[f"{name}_{aux_name}"] = aux_loss
            
            # Collect metrics
            if component_output.metrics:
                for metric_name, metric_value in component_output.metrics.items():
                    metrics[f"{name}_{metric_name}"] = metric_value
            
            # Collect debug info
            if component_output.debug_info:
                debug_info[name] = component_output.debug_info
            
            # Track individual component losses
            metrics[f"{name}_loss"] = component_output.loss.item()
        
        return LossOutput(
            loss=total_loss,
            auxiliary_losses=auxiliary_losses if auxiliary_losses else None,
            metrics=metrics if metrics else None,
            debug_info=debug_info if debug_info else None
        )
    
    def add_component(self, name: str, component: BaseLoss):
        """Add a new loss component."""
        self.loss_components[name] = component
        if self.normalize_weights:
            self._normalize_component_weights()
    
    def remove_component(self, name: str):
        """Remove a loss component."""
        if name in self.loss_components:
            del self.loss_components[name]
            if self.normalize_weights:
                self._normalize_component_weights()
    
    def set_component_weight(self, name: str, weight: float):
        """Set weight for a specific component."""
        if name in self.loss_components:
            self.loss_components[name].weight = weight
            if self.normalize_weights:
                self._normalize_component_weights()

class AdaptiveLoss(BaseLoss):
    """
    Loss function that adapts its behavior based on training progress.
    
    This class provides a framework for implementing losses that
    change their behavior during training, such as curriculum learning
    or progressive difficulty adjustment.
    """
    
    def __init__(
        self,
        adaptation_schedule: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize adaptive loss.
        
        Args:
            adaptation_schedule: Configuration for loss adaptation
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.adaptation_schedule = adaptation_schedule
        self.adaptation_milestones = sorted(adaptation_schedule.get("milestones", []))
        self.current_phase = 0
    
    def _update_adaptation_phase(self):
        """Update the current adaptation phase based on training step."""
        while (self.current_phase < len(self.adaptation_milestones) and 
               self.current_step >= self.adaptation_milestones[self.current_phase]):
            self.current_phase += 1
            self._on_phase_change()
    
    def _on_phase_change(self):
        """Called when adaptation phase changes. Override in subclasses."""
        pass
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Compute adaptive loss."""
        self._update_adaptation_phase()
        return self._compute_adaptive_loss(logits, labels, **kwargs)
    
    @abstractmethod
    def _compute_adaptive_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Implement adaptive loss computation in subclasses."""
        pass