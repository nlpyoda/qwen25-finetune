# Getting Started with Qwen2.5 Training Framework

This guide will help you get up and running with the Qwen2.5 Training Framework quickly.

## 🚀 Quick Start (5 minutes)

### 1. Installation

```bash
git clone https://github.com/your-username/qwen25-training-framework.git
cd qwen25-training-framework

# Install dependencies
./scripts/setup_environment.sh

# Or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.47.0 accelerate>=1.1.0 peft>=0.15.0 datasets
pip install deepspeed>=0.16.0 flash-attn>=2.0.0 liger-kernel>=0.5.0
pip install qwen-vl-utils Pillow opencv-python trl>=0.17.0
```

### 2. Basic Training

```bash
# Create sample data
python examples/create_sample_data.py

# Train with default configuration
python train.py --config configs/qwen25_7b_lora.yaml

# Monitor training
tensorboard --logdir ./output/runs
```

### 3. Multi-GPU Training

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=8 train.py --config configs/distributed/single_node.yaml

# Multi-node training (see Multi-Node Setup section)
```

## 📋 Detailed Setup

### Environment Requirements

- **Python**: 3.10+
- **CUDA**: 12.1+ (for GPU training)
- **Memory**: 32GB+ RAM, 80GB+ GPU memory for 7B model
- **Storage**: 100GB+ for model and data

### Installing from Source

```bash
# Clone repository
git clone https://github.com/your-username/qwen25-training-framework.git
cd qwen25-training-framework

# Create conda environment
conda create -n qwen25-training python=3.10
conda activate qwen25-training

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Verify installation
python scripts/validate_installation.py
```

### Validate Installation

```bash
python scripts/validate_installation.py
```

Expected output:
```
✅ CUDA available: True (8 devices)
✅ Transformers version: 4.47.0+
✅ All imports successful
✅ Qwen2.5 model accessible
✅ Framework components working
```

## 🎯 Training Your First Model

### Step 1: Prepare Your Data

Create a JSONL file with conversation data:

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thanks!"}]}
```

### Step 2: Create Configuration

Copy and modify a base configuration:

```bash
cp configs/qwen25_7b_lora.yaml my_training_config.yaml
```

Edit `my_training_config.yaml`:

```yaml
# Model configuration
model:
  architecture: "qwen25_vl"
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"

# Data configuration  
data:
  train_file: "path/to/your/train.jsonl"
  validation_file: "path/to/your/val.jsonl"
  max_seq_length: 2048

# Training configuration
training:
  output_dir: "./my_model_output"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2e-5
  bf16: true
```

### Step 3: Start Training

```bash
# Single GPU
python train.py --config my_training_config.yaml

# Multiple GPUs
torchrun --nproc_per_node=8 train.py --config my_training_config.yaml
```

### Step 4: Monitor Progress

```bash
# View logs
tail -f my_model_output/training.log

# TensorBoard
tensorboard --logdir my_model_output/runs

# Weights & Biases (if configured)
wandb login
# Check your W&B dashboard
```

## 🏗️ Customization Examples

### Adding Special Tokens

```yaml
special_tokens:
  tokens:
    - token: "<CUSTOM_TASK>"
      description: "Custom task marker"
      trainable: true
    - token: "<EXPERT_MODE>"
      description: "Expert mode indicator"
      trainable: true
```

### Using Custom Loss Functions

```yaml
loss:
  type: "focal"
  config:
    alpha: 1.0
    gamma: 2.0
    weight: 1.0
```

### Multi-Component Loss

```yaml
loss:
  - name: "main_loss"
    type: "label_smoothing"
    weight: 1.0
    config:
      smoothing: 0.1
  - name: "auxiliary_loss"
    type: "contrastive"
    weight: 0.1
    config:
      temperature: 0.07
```

### Custom Architecture

```python
# In your custom architecture file
from src.models.base import QwenVLArchitecture
from src.models.factory import register_architecture

class MyCustomQwen(QwenVLArchitecture):
    def _init_custom_layers(self):
        # Add your custom layers
        pass
    
    def _apply_custom_forward(self, hidden_states, **kwargs):
        # Custom forward logic
        return hidden_states

# Register architecture
register_architecture("my_custom_qwen", MyCustomQwen)
```

Then in config:
```yaml
model:
  architecture: "my_custom_qwen"
  config:
    # Custom config parameters
```

## 🌐 Multi-Node Training Setup

### AWS p4d.24xlarge Setup

```bash
# 1. Launch instances
python scripts/launch_aws_cluster.py --instance-type p4d.24xlarge --nodes 2

# 2. Setup each node
./scripts/setup_node.sh

# 3. Start training on master node (rank 0)
torchrun --nnodes=2 --node_rank=0 --master_addr=$MASTER_IP --master_port=29500 \
         --nproc_per_node=8 train.py --config configs/distributed/multi_node_p4d.yaml

# 4. Start training on worker nodes (rank 1+)  
torchrun --nnodes=2 --node_rank=1 --master_addr=$MASTER_IP --master_port=29500 \
         --nproc_per_node=8 train.py --config configs/distributed/multi_node_p4d.yaml
```

### Using the Launcher Script

```bash
# Auto-detect and launch
python scripts/launch_distributed.py --auto train.py --config my_config.yaml

# Manual configuration
python scripts/launch_distributed.py \
    --nodes 2 --node-rank 0 --master-addr 10.0.1.100 \
    train.py --config my_config.yaml

# Dry run to see command
python scripts/launch_distributed.py --auto --dry-run train.py --config my_config.yaml
```

## 🔧 Configuration Reference

### Model Configurations

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| `qwen25_vl` | Standard Qwen2.5-VL | General vision-language tasks |
| `custom_qwen_vl` | Customizable base | Custom architectures |
| `memory_augmented_qwen` | With external memory | Long-context tasks |

### Loss Functions

| Loss Type | Description | Parameters |
|-----------|-------------|------------|
| `label_smoothing` | Smoothed cross-entropy | `smoothing` |
| `focal` | Focal loss for imbalanced data | `alpha`, `gamma` |
| `contrastive` | Contrastive learning | `temperature` |
| `kl_divergence` | Knowledge distillation | `temperature` |

### Training Modes

| Mode | Description | Configuration |
|------|-------------|--------------|
| Full Fine-tuning | Train all parameters | `lora_enable: false` |
| LoRA | Low-rank adaptation | `lora_enable: true` |
| QLoRA | Quantized LoRA | `bits: 4`, `lora_enable: true` |

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: true

# Use mixed precision
bf16: true
```

**Slow Training**
```bash
# Increase data workers
dataloader_num_workers: 16

# Use faster attention
# (automatically enabled with flash-attn)

# Optimize data loading
pin_memory: true
```

**Distributed Training Issues**
```bash
# Check network connectivity
python scripts/test_connectivity.py

# Verify NCCL setup
python -c "import torch; print(torch.distributed.is_nccl_available())"

# Check environment variables
env | grep -E "(MASTER_ADDR|MASTER_PORT|WORLD_SIZE|RANK)"
```

### Getting Help

1. **Check logs**: Look in `output_dir/training.log`
2. **Run validation**: `python scripts/validate_installation.py`
3. **Test examples**: `python examples/complete_training_example.py`
4. **Check configs**: Validate YAML syntax and parameters

### Performance Tips

1. **Use appropriate batch sizes**: Start small and increase gradually
2. **Enable mixed precision**: Use `bf16: true` for A100 GPUs
3. **Optimize data loading**: Use multiple workers and pin memory
4. **Monitor GPU utilization**: Use `nvidia-smi` or `nvtop`
5. **Profile your training**: Use PyTorch profiler for bottlenecks

## 📚 Next Steps

- **Advanced Features**: See `docs/advanced_features.md`
- **Custom Architectures**: Check `examples/custom_architecture_example.py`
- **Production Deployment**: Read `docs/production_deployment.md`
- **API Reference**: Browse `docs/api_reference.md`

## 💡 Tips for Success

1. **Start small**: Begin with a small model and dataset
2. **Monitor closely**: Watch loss curves and GPU utilization
3. **Save frequently**: Use appropriate `save_steps` values
4. **Experiment systematically**: Change one parameter at a time
5. **Use version control**: Track your configurations and results

Happy training! 🚀