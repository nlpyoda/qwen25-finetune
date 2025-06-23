"""
Complete Training Example

This example demonstrates a full training pipeline using the framework's
features: custom architecture, special tokens, custom loss, and distributed training.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.factory import model_factory, create_model, create_tokenizer
from losses.factory import loss_factory, create_loss
from data.tokenizers.special_tokens import SpecialTokenManager
from utils.distributed import DistributedManager, DistributedConfigBuilder

def create_training_config() -> Dict[str, Any]:
    """
    Create a comprehensive training configuration that showcases
    all framework features.
    """
    config = {
        # Model configuration with custom architecture
        "model": {
            "architecture": "qwen25_vl",  # Can be changed to custom architectures
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "config": {
                "gradient_checkpointing": True,
                "use_cache": False,
                "torch_dtype": "bfloat16"
            }
        },
        
        # Tokenizer configuration
        "tokenizer": {
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "trust_remote_code": True
        },
        
        # Special tokens configuration
        "special_tokens": {
            "tokens": [
                {
                    "token": "<SYSTEM>",
                    "description": "System message prefix",
                    "trainable": True,
                    "is_system_token": True
                },
                {
                    "token": "<USER>",
                    "description": "User message prefix", 
                    "trainable": True,
                    "is_user_token": True
                },
                {
                    "token": "<ASSISTANT>",
                    "description": "Assistant response prefix",
                    "trainable": True
                },
                {
                    "token": "<TOOL_CALL>",
                    "description": "Function call start",
                    "trainable": True,
                    "is_function_token": True
                },
                {
                    "token": "</TOOL_CALL>",
                    "description": "Function call end",
                    "trainable": True,
                    "is_function_token": True
                },
                {
                    "token": "<VISION>",
                    "description": "Vision content marker",
                    "trainable": True,
                    "is_vision_token": True
                }
            ]
        },
        
        # Multi-component loss configuration
        "loss": [
            {
                "name": "main_loss",
                "type": "label_smoothing",
                "weight": 1.0,
                "config": {
                    "smoothing": 0.1
                }
            },
            {
                "name": "focal_loss", 
                "type": "focal",
                "weight": 0.1,
                "config": {
                    "alpha": 1.0,
                    "gamma": 2.0
                }
            }
        ],
        
        # Training configuration
        "training": {
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 10,
            "dataloader_num_workers": 8,
            "remove_unused_columns": False,
            "report_to": ["tensorboard"],
            "bf16": True,
            "gradient_checkpointing": True
        },
        
        # Data configuration
        "data": {
            "train_file": "data/train.jsonl",
            "validation_file": "data/validation.jsonl",
            "max_seq_length": 2048,
            "preprocessing_num_workers": 8
        },
        
        # Distributed training configuration
        "distributed": {
            "backend": "nccl",
            "nodes": 1,
            "node_rank": 0,
            "world_size": 8,
            "master_addr": "localhost",
            "master_port": 29500,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True
        },
        
        # Monitoring configuration
        "monitoring": {
            "wandb_project": "qwen25-training-framework",
            "log_model_architecture": True,
            "save_model_every_epoch": True
        }
    }
    
    return config

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """
    Setup model and tokenizer with special tokens.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (model, tokenizer, token_manager)
    """
    print("🏗️  Setting up model and tokenizer...")
    
    # Create tokenizer
    tokenizer = create_tokenizer(config_dict=config)
    print(f"✅ Loaded tokenizer: {len(tokenizer)} tokens")
    
    # Setup special tokens
    token_manager = SpecialTokenManager(tokenizer)
    
    # Add special tokens from config
    special_tokens_config = config.get("special_tokens", {})
    tokens_to_add = special_tokens_config.get("tokens", [])
    
    if tokens_to_add:
        token_ids = token_manager.add_tokens(tokens_to_add)
        print(f"✅ Added {len(token_ids)} special tokens")
        
        # Print token summary
        summary = token_manager.summary
        print(f"📊 Token summary: {summary}")
    
    # Create model
    model = create_model(config_dict=config)
    print(f"✅ Loaded model: {model.__class__.__name__}")
    
    # Resize model embeddings for new tokens
    if tokens_to_add:
        token_manager.resize_model_embeddings(model)
        print(f"✅ Resized model embeddings to {len(tokenizer)} tokens")
    
    return model, tokenizer, token_manager

def setup_loss_function(config: Dict[str, Any]):
    """
    Setup custom loss function from configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured loss function
    """
    print("📊 Setting up loss function...")
    
    loss_fn = create_loss(config_dict=config)
    print(f"✅ Created loss function: {loss_fn.__class__.__name__}")
    
    return loss_fn

def setup_distributed_training(config: Dict[str, Any]):
    """
    Setup distributed training configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        DistributedManager instance
    """
    print("🌐 Setting up distributed training...")
    
    # Build distributed config
    dist_config = (DistributedConfigBuilder()
                  .from_yaml("temp_config.yaml")  # Would load from actual config
                  .from_env()  # Override with environment variables
                  .build())
    
    # Create distributed manager
    dist_manager = DistributedManager(dist_config)
    
    # Initialize distributed training
    if dist_manager.setup():
        print(f"✅ Distributed training initialized")
        print(f"   World size: {dist_manager.world_size}")
        print(f"   Local rank: {dist_manager.local_rank}")
        print(f"   Is main process: {dist_manager.is_main_process}")
    else:
        print("⚠️  Failed to initialize distributed training, using single GPU")
        dist_manager = None
    
    return dist_manager

def create_sample_data():
    """Create sample training data in the expected format."""
    print("📝 Creating sample training data...")
    
    # Sample conversation data with special tokens
    sample_data = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can process both text and vision inputs."
                },
                {
                    "role": "user", 
                    "content": "What do you see in this image? <VISION>placeholder_for_image</VISION>"
                },
                {
                    "role": "assistant",
                    "content": "I can see an image has been provided. Let me analyze it for you."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Can you help me calculate 15 * 23?"
                },
                {
                    "role": "assistant", 
                    "content": "<TOOL_CALL>calculate(15 * 23)</TOOL_CALL>The result is 345."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in mathematics and science."
                },
                {
                    "role": "user",
                    "content": "Explain quantum entanglement in simple terms."
                },
                {
                    "role": "assistant",
                    "content": "Quantum entanglement is a phenomenon where two particles become connected in such a way that measuring one instantly affects the other, regardless of distance."
                }
            ]
        }
    ]
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save training data
    import json
    train_file = data_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    # Create validation data (same for demo)
    val_file = data_dir / "validation.jsonl"
    with open(val_file, 'w') as f:
        f.write(json.dumps(sample_data[0]) + '\n')
    
    print(f"✅ Created sample data: {train_file}, {val_file}")
    return train_file, val_file

def format_conversation_with_tokens(messages, tokenizer):
    """
    Format conversation messages with special tokens.
    
    Args:
        messages: List of message dictionaries
        tokenizer: Tokenizer with special tokens
        
    Returns:
        Formatted text string
    """
    formatted_text = ""
    
    for message in messages:
        role = message["role"].upper()
        content = message["content"]
        
        # Use special tokens if available
        if f"<{role}>" in tokenizer.vocab:
            formatted_text += f"<{role}>{content}"
        else:
            formatted_text += f"{role}: {content}\n"
    
    return formatted_text.strip()

def run_training_example():
    """
    Run a complete training example showcasing all framework features.
    """
    print("🚀 Qwen2.5 Training Framework - Complete Example")
    print("=" * 60)
    
    # Create configuration
    config = create_training_config()
    
    # Save config to temporary file for distributed setup
    with open("temp_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Setup components
        model, tokenizer, token_manager = setup_model_and_tokenizer(config)
        loss_fn = setup_loss_function(config)
        dist_manager = setup_distributed_training(config)
        
        # Create sample data
        train_file, val_file = create_sample_data()
        
        # Wrap model for distributed training if available
        if dist_manager:
            model = dist_manager.wrap_model(model)
        
        # Example training step
        print("\n🔄 Example training step...")
        
        # Format sample conversation
        sample_conversation = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
            ]
        }
        
        formatted_text = format_conversation_with_tokens(
            sample_conversation["messages"], tokenizer
        )
        print(f"Formatted conversation: '{formatted_text}'")
        
        # Tokenize
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        print(f"Model output shape: {outputs.logits.shape}")
        print(f"Loss: {outputs.loss.item():.4f}")
        
        # Test custom loss
        loss_output = loss_fn(outputs.logits, inputs["input_ids"])
        print(f"Custom loss: {loss_output.loss.item():.4f}")
        
        if loss_output.auxiliary_losses:
            print("Auxiliary losses:")
            for name, aux_loss in loss_output.auxiliary_losses.items():
                print(f"  {name}: {aux_loss.item():.6f}")
        
        # Show training metrics
        if loss_output.metrics:
            print("Training metrics:")
            for name, metric in loss_output.metrics.items():
                print(f"  {name}: {metric}")
        
        # Export configurations
        print("\n💾 Exporting configurations...")
        
        # Export token configuration
        token_manager.export_config("token_config.yaml")
        
        # Export model architecture info
        if hasattr(model, 'architecture_info'):
            arch_info = model.architecture_info
            with open("architecture_info.yaml", 'w') as f:
                yaml.dump(arch_info, f, default_flow_style=False)
        
        print("✅ Training example completed successfully!")
        
        # Print usage instructions
        print("\n📚 To use this configuration for actual training:")
        print("1. Prepare your dataset in JSONL format")
        print("2. Adjust the configuration in temp_config.yaml")
        print("3. Run: python train.py --config temp_config.yaml")
        print("4. For distributed training:")
        print("   torchrun --nproc_per_node=8 train.py --config temp_config.yaml")
        
    except Exception as e:
        print(f"❌ Error in training example: {e}")
        raise
    
    finally:
        # Cleanup
        if dist_manager:
            dist_manager.cleanup()
        
        # Clean up temporary files
        temp_files = ["temp_config.yaml"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    """Main entry point for the complete training example."""
    try:
        run_training_example()
    except KeyboardInterrupt:
        print("\n⚠️  Training example interrupted by user")
    except Exception as e:
        print(f"\n❌ Training example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()