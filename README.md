# Qwen2.5 Training Framework

A modular, extensible framework for training Qwen2.5 models with support for custom architectures, special tokens, and loss functions.

## 🚀 Features

- **🏗️ Modular Architecture**: Easy to extend and customize model architectures
- **🎯 Custom Loss Functions**: Plug-and-play loss function system
- **🔤 Special Tokens**: Simple system for adding custom tokens
- **⚡ Multi-node Training**: Optimized for p4.24xlarge instances with NCCL
- **🔧 Both LoRA & Full Fine-tuning**: Flexible training modes
- **📊 Comprehensive Logging**: Built-in monitoring and visualization
- **🧪 Testing Suite**: Incremental testing from single GPU to multi-node

## 📋 Quick Start

### Installation

```bash
git clone https://github.com/your-username/qwen25-training-framework.git
cd qwen25-training-framework
./scripts/setup_environment.sh
```

### Basic Training

```bash
# Single node training
python train.py --config configs/qwen25_7b_lora.yaml

# Multi-node training (master node)
./scripts/launch_multinode.sh --config configs/qwen25_7b_multinode.yaml --node-rank 0 --master-addr MASTER_IP

# Multi-node training (worker nodes)
./scripts/launch_multinode.sh --config configs/qwen25_7b_multinode.yaml --node-rank 1 --master-addr MASTER_IP
```

## 🏗️ Architecture

```
qwen25-training-framework/
├── 📂 configs/                    # Training configurations
│   ├── models/                    # Model architecture configs
│   ├── losses/                    # Loss function configs  
│   └── tokens/                    # Special token configs
├── 📂 src/
│   ├── 📂 models/                 # Model architectures
│   │   ├── architectures/         # Custom architecture implementations
│   │   ├── adapters/              # LoRA and other adapters
│   │   └── factory.py             # Model factory for easy extension
│   ├── 📂 losses/                 # Loss functions
│   │   ├── custom_losses.py       # Custom loss implementations
│   │   └── loss_factory.py        # Loss factory
│   ├── 📂 data/                   # Data handling
│   │   ├── tokenizers/            # Token management
│   │   └── datasets.py            # Dataset implementations
│   ├── 📂 training/               # Training logic
│   │   ├── trainer.py             # Main trainer
│   │   └── callbacks.py           # Training callbacks
│   └── 📂 utils/                  # Utilities
│       ├── distributed.py         # Multi-node utilities
│       └── logging.py             # Logging utilities
├── 📂 scripts/                    # Setup and launch scripts
├── 📂 tests/                      # Testing suite
└── 📂 examples/                   # Usage examples
```

## 🎯 Custom Features

### Adding Custom Architectures

Create a new architecture in `src/models/architectures/`:

```python
from src.models.base import BaseQwenArchitecture

class MyCustomQwen(BaseQwenArchitecture):
    def __init__(self, config):
        super().__init__(config)
        # Your custom layers here
        
    def forward(self, *args, **kwargs):
        # Your custom forward pass
        pass
```

Register it in `configs/models/my_custom.yaml`:

```yaml
model:
  architecture: "MyCustomQwen"
  config:
    # Your config here
```

### Adding Special Tokens

Define tokens in `configs/tokens/my_tokens.yaml`:

```yaml
special_tokens:
  - token: "<CUSTOM_TOKEN>"
    description: "Custom functionality token"
    trainable: true
  - token: "<SYSTEM>"
    description: "System message token"
    trainable: false
```

### Adding Custom Loss Functions

Create a loss in `src/losses/custom_losses.py`:

```python
from src.losses.base import BaseLoss

class MyCustomLoss(BaseLoss):
    def forward(self, logits, labels, **kwargs):
        # Your custom loss logic
        return loss
```

Register it in `configs/losses/my_loss.yaml`:

```yaml
loss:
  type: "MyCustomLoss"
  weight: 1.0
  config:
    # Loss-specific config
```

## 📊 Monitoring

- **TensorBoard**: Automatic logging of metrics
- **Weights & Biases**: Optional W&B integration
- **Custom Metrics**: Easy to add domain-specific metrics

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_models.py
python -m pytest tests/test_losses.py

# Integration tests
./scripts/test_training.sh
```

## 🔧 Configuration

All training is controlled via YAML configs. See `configs/` for examples:

- `qwen25_7b_lora.yaml` - LoRA training
- `qwen25_7b_full.yaml` - Full fine-tuning  
- `qwen25_7b_multinode.yaml` - Multi-node training
- `custom_architecture.yaml` - Custom architecture example

## 📚 Examples

See `examples/` for:
- Custom architecture implementation
- Special token usage
- Custom loss function
- Multi-modal training setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.