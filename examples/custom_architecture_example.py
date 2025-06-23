"""
Example: Creating a Custom Qwen2.5 Architecture

This example demonstrates how to create a custom architecture that extends
the base Qwen2.5 model with additional layers and functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Import the framework components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.base import BaseQwenArchitecture, BaseQwenConfig, QwenVLArchitecture
from models.factory import register_architecture
from losses.base import BaseLoss, LossOutput

class MemoryAugmentedQwenConfig(BaseQwenConfig):
    """
    Configuration for Memory-Augmented Qwen architecture.
    
    This extends the base configuration with memory-specific parameters.
    """
    
    def __init__(
        self,
        memory_size: int = 1024,
        memory_heads: int = 8,
        memory_temperature: float = 1.0,
        enable_memory_gate: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_heads = memory_heads
        self.memory_temperature = memory_temperature
        self.enable_memory_gate = enable_memory_gate

class MemoryAugmentedQwen(QwenVLArchitecture):
    """
    Memory-Augmented Qwen Architecture.
    
    This custom architecture adds an external memory mechanism that allows
    the model to store and retrieve information across sequences. This is
    useful for tasks requiring long-term memory or knowledge retrieval.
    
    Key Features:
    - External memory bank with attention-based retrieval
    - Memory gate to control when to use external vs internal memory
    - Compatible with vision-language inputs
    - Configurable memory size and attention heads
    
    Example Usage:
        ```python
        config = MemoryAugmentedQwenConfig(
            memory_size=2048,
            memory_heads=16,
            vocab_size=152064
        )
        
        model = MemoryAugmentedQwen(config)
        
        # Training with memory
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_memory=True
        )
        ```
    """
    
    config_class = MemoryAugmentedQwenConfig
    
    def _init_custom_layers(self):
        """Initialize memory-specific layers."""
        config = self.config
        
        # External memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(config.memory_size, config.hidden_size) * 0.02
        )
        
        # Memory attention mechanism
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.memory_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Memory gate (controls when to use external memory)
        if config.enable_memory_gate:
            self.memory_gate = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        # Output projection for memory-augmented states
        self.memory_projection = nn.Linear(
            config.hidden_size * 2,  # concatenated memory + hidden states
            config.hidden_size
        )
        
        # Layer normalization for memory outputs
        self.memory_norm = nn.LayerNorm(config.hidden_size)
        
        print(f"✅ Initialized Memory-Augmented Qwen with {config.memory_size} memory slots")
    
    def _apply_custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply memory-augmented forward pass.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            use_memory: Whether to use external memory
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing the memory-augmented hidden states and memory weights
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if not use_memory:
            # Skip memory augmentation
            return (hidden_states, None)
        
        # Retrieve from external memory
        memory_output, memory_weights = self._retrieve_from_memory(
            hidden_states, attention_mask
        )
        
        # Apply memory gate if enabled
        if hasattr(self, 'memory_gate'):
            gate_values = self.memory_gate(hidden_states)  # [batch, seq_len, 1]
            memory_output = memory_output * gate_values
        
        # Combine hidden states with memory
        combined_states = torch.cat([hidden_states, memory_output], dim=-1)
        
        # Project back to original dimension
        augmented_states = self.memory_projection(combined_states)
        augmented_states = self.memory_norm(augmented_states)
        
        # Residual connection
        output_states = hidden_states + augmented_states
        
        return (output_states, memory_weights)
    
    def _retrieve_from_memory(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant information from external memory.
        
        Args:
            hidden_states: Query states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (retrieved_memory, attention_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        memory_size = self.memory_bank.size(0)
        
        # Expand memory bank for batch
        memory_keys = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        memory_values = memory_keys  # Use same tensor as keys and values
        
        # Reshape queries for attention
        queries = hidden_states
        
        # Apply memory attention
        retrieved_memory, attention_weights = self.memory_attention(
            query=queries,
            key=memory_keys,
            value=memory_values,
            need_weights=True
        )
        
        return retrieved_memory, attention_weights
    
    def update_memory(
        self,
        hidden_states: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        learning_rate: float = 0.01
    ):
        """
        Update external memory based on current hidden states.
        
        This method allows the model to write new information to memory
        during training or inference.
        
        Args:
            hidden_states: States to write to memory [batch_size, seq_len, hidden_size]
            memory_mask: Mask for which memory slots to update
            learning_rate: Learning rate for memory updates
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute similarity between hidden states and memory
        similarities = torch.matmul(
            hidden_states.view(-1, hidden_size),  # [batch*seq, hidden]
            self.memory_bank.t()  # [hidden, memory_size]
        )  # [batch*seq, memory_size]
        
        # Get indices of most similar memory slots
        _, memory_indices = similarities.max(dim=-1)  # [batch*seq]
        
        # Update memory with exponential moving average
        with torch.no_grad():
            for i, idx in enumerate(memory_indices):
                current_state = hidden_states.view(-1, hidden_size)[i]
                self.memory_bank[idx] = (
                    (1 - learning_rate) * self.memory_bank[idx] +
                    learning_rate * current_state
                )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current memory state."""
        with torch.no_grad():
            memory_norms = torch.norm(self.memory_bank, dim=-1)
            
            return {
                "memory_size": self.memory_bank.size(0),
                "memory_dim": self.memory_bank.size(1),
                "avg_memory_norm": memory_norms.mean().item(),
                "max_memory_norm": memory_norms.max().item(),
                "min_memory_norm": memory_norms.min().item(),
                "memory_std": memory_norms.std().item()
            }
    
    def reset_memory(self):
        """Reset memory bank to random initialization."""
        with torch.no_grad():
            nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)
        print("🔄 Memory bank reset")

class MemoryAugmentedLoss(BaseLoss):
    """
    Custom loss function for memory-augmented models.
    
    This loss combines standard language modeling loss with
    memory-specific regularization terms.
    """
    
    def __init__(
        self,
        memory_l2_weight: float = 0.01,
        memory_sparsity_weight: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory_l2_weight = memory_l2_weight
        self.memory_sparsity_weight = memory_sparsity_weight
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        memory_weights: Optional[torch.Tensor] = None,
        memory_bank: Optional[torch.Tensor] = None,
        **kwargs
    ) -> LossOutput:
        """Compute memory-augmented loss."""
        
        # Standard language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        
        auxiliary_losses = {}
        metrics = {"language_modeling_loss": lm_loss.item()}
        
        # Memory regularization losses
        if memory_bank is not None:
            # L2 regularization on memory bank
            memory_l2_loss = torch.norm(memory_bank, p=2) * self.memory_l2_weight
            auxiliary_losses["memory_l2"] = memory_l2_loss
            metrics["memory_l2_loss"] = memory_l2_loss.item()
        
        if memory_weights is not None:
            # Sparsity regularization on memory attention
            memory_entropy = -torch.sum(
                memory_weights * torch.log(memory_weights + 1e-8), dim=-1
            ).mean()
            memory_sparsity_loss = memory_entropy * self.memory_sparsity_weight
            auxiliary_losses["memory_sparsity"] = memory_sparsity_loss
            metrics["memory_sparsity_loss"] = memory_sparsity_loss.item()
            metrics["avg_memory_attention_entropy"] = memory_entropy.item()
        
        return LossOutput(
            loss=lm_loss,
            auxiliary_losses=auxiliary_losses if auxiliary_losses else None,
            metrics=metrics
        )

def create_memory_augmented_model_example():
    """
    Example function showing how to create and use a memory-augmented model.
    """
    print("🏗️  Creating Memory-Augmented Qwen Model Example")
    print("=" * 60)
    
    # 1. Create configuration
    config = MemoryAugmentedQwenConfig(
        memory_size=512,
        memory_heads=8,
        memory_temperature=1.0,
        enable_memory_gate=True,
        hidden_size=4096,
        vocab_size=152064
    )
    
    # 2. Create model
    model = MemoryAugmentedQwen(config)
    
    # 3. Print model info
    print(f"📊 Model Information:")
    print(f"   Architecture: {model.config.architecture_type}")
    print(f"   Memory size: {config.memory_size}")
    print(f"   Memory heads: {config.memory_heads}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Example forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    print(f"\n🔄 Example forward pass:")
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass with memory
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_memory=True
        )
    
    print(f"   Output shape: {outputs.logits.shape}")
    print(f"   Loss: {outputs.loss.item():.4f}")
    
    # 5. Memory statistics
    memory_stats = model.get_memory_statistics()
    print(f"\n📈 Memory Statistics:")
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 6. Create custom loss
    memory_loss = MemoryAugmentedLoss(
        memory_l2_weight=0.01,
        memory_sparsity_weight=0.001
    )
    
    print(f"\n📊 Custom Loss Example:")
    loss_output = memory_loss(
        logits=outputs.logits,
        labels=labels,
        memory_bank=model.memory_bank,
        memory_weights=torch.rand(batch_size, seq_len, config.memory_size)  # Mock weights
    )
    
    print(f"   Total loss: {loss_output.loss.item():.4f}")
    if loss_output.auxiliary_losses:
        for name, aux_loss in loss_output.auxiliary_losses.items():
            print(f"   {name}: {aux_loss.item():.6f}")
    
    return model, memory_loss

def register_memory_augmented_architecture():
    """Register the memory-augmented architecture with the model factory."""
    register_architecture("memory_augmented_qwen", MemoryAugmentedQwen)
    print("✅ Memory-Augmented Qwen architecture registered")

if __name__ == "__main__":
    # Register the architecture
    register_memory_augmented_architecture()
    
    # Create and demonstrate the model
    model, loss_fn = create_memory_augmented_model_example()
    
    print(f"\n🎉 Memory-Augmented Qwen Example Complete!")
    print(f"\nTo use this architecture in training:")
    print(f"1. Add 'memory_augmented_qwen' to your model config")
    print(f"2. Use the MemoryAugmentedLoss for training")
    print(f"3. Configure memory-specific hyperparameters")
    
    # Example configuration
    example_config = {
        "model": {
            "architecture": "memory_augmented_qwen",
            "config": {
                "memory_size": 1024,
                "memory_heads": 16,
                "memory_temperature": 1.0,
                "enable_memory_gate": True
            }
        },
        "loss": {
            "type": "custom",
            "class": "MemoryAugmentedLoss",
            "config": {
                "memory_l2_weight": 0.01,
                "memory_sparsity_weight": 0.001
            }
        }
    }
    
    print(f"\nExample YAML configuration:")
    import yaml
    print(yaml.dump(example_config, default_flow_style=False))