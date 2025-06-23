"""
Custom trainer for Qwen2.5 models.
"""

import torch
from transformers import Trainer
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QwenTrainer(Trainer):
    """
    Custom trainer for Qwen2.5 models with framework integration.
    """
    
    def __init__(self, custom_loss_fn=None, **kwargs):
        """
        Initialize custom trainer.
        
        Args:
            custom_loss_fn: Optional custom loss function
            **kwargs: Arguments passed to base Trainer
        """
        super().__init__(**kwargs)
        self.custom_loss_fn = custom_loss_fn
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss using custom loss function if provided.
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        if self.custom_loss_fn is not None and labels is not None:
            # Use custom loss function
            loss_output = self.custom_loss_fn(outputs.logits, labels)
            loss = loss_output.loss
            
            # Log additional metrics if available
            if loss_output.metrics:
                for key, value in loss_output.metrics.items():
                    if isinstance(value, (int, float)):
                        self.log({f"train/{key}": value})
        else:
            # Use default loss
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss