#!/usr/bin/env python3
"""
Main training script for Qwen2.5 Training Framework.

This script provides the main entry point for training Qwen2.5 models
with custom architectures, loss functions, and distributed training.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.factory import create_model, create_tokenizer
from losses.factory import create_loss
from data.tokenizers.special_tokens import SpecialTokenManager
from utils.distributed import DistributedManager, create_distributed_config_from_yaml
from utils.logging import setup_logging

# Setup logging
logger = logging.getLogger(__name__)

class Qwen25Trainer:
    """
    Main trainer class for Qwen2.5 models.
    
    This class orchestrates the training process, handling model creation,
    data loading, distributed training setup, and training execution.
    """
    
    def __init__(self, config_path: str, **kwargs):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to training configuration YAML file
            **kwargs: Additional configuration overrides
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Apply any command-line overrides
        self._apply_overrides(kwargs)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.token_manager = None
        self.loss_fn = None
        self.dist_manager = None
        self.trainer = None
        
        # Setup logging
        setup_logging(
            level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file")
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply command-line configuration overrides."""
        for key, value in overrides.items():
            if key in self.config:
                logger.info(f"Overriding config: {key} = {value}")
                self.config[key] = value
    
    def setup_distributed(self):
        """Setup distributed training if configured."""
        distributed_config = self.config.get("distributed")
        
        if distributed_config and distributed_config.get("nodes", 1) > 1:
            logger.info("Setting up distributed training...")
            
            # Create distributed config
            dist_config = create_distributed_config_from_yaml(self.config_path)
            
            # Override with environment variables
            dist_config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist_config.node_rank = int(os.environ.get("RANK", 0)) // torch.cuda.device_count()
            
            # Create and setup distributed manager
            self.dist_manager = DistributedManager(dist_config)
            
            if not self.dist_manager.setup():
                raise RuntimeError("Failed to setup distributed training")
            
            logger.info("Distributed training setup complete")
        else:
            logger.info("Single-node training mode")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with special tokens."""
        logger.info("Setting up model and tokenizer...")
        
        # Create tokenizer
        self.tokenizer = create_tokenizer(config_dict=self.config)
        logger.info(f"Loaded tokenizer with {len(self.tokenizer)} tokens")
        
        # Setup special tokens
        special_tokens_config = self.config.get("special_tokens")
        if special_tokens_config:
            self.token_manager = SpecialTokenManager(self.tokenizer)
            
            tokens_to_add = special_tokens_config.get("tokens", [])
            if tokens_to_add:
                token_ids = self.token_manager.add_tokens(tokens_to_add)
                logger.info(f"Added {len(token_ids)} special tokens")
        
        # Create model
        self.model = create_model(config_dict=self.config)
        logger.info(f"Loaded model: {self.model.__class__.__name__}")
        
        # Resize embeddings if special tokens were added
        if self.token_manager and special_tokens_config.get("tokens"):
            self.token_manager.resize_model_embeddings(self.model)
            logger.info(f"Resized embeddings to {len(self.tokenizer)} tokens")
        
        # Wrap model for distributed training
        if self.dist_manager:
            self.model = self.dist_manager.wrap_model(self.model)
            logger.info("Model wrapped for distributed training")
    
    def setup_loss_function(self):
        """Setup loss function from configuration."""
        logger.info("Setting up loss function...")
        
        self.loss_fn = create_loss(config_dict=self.config)
        logger.info(f"Created loss function: {self.loss_fn.__class__.__name__}")
    
    def create_data_module(self):
        """Create data module for training."""
        from data.datasets import create_dataset
        
        logger.info("Creating data module...")
        
        data_config = self.config.get("data", {})
        
        # Create datasets
        train_dataset = create_dataset(
            data_path=data_config.get("train_file"),
            tokenizer=self.tokenizer,
            max_length=data_config.get("max_seq_length", 2048),
            mode="train"
        )
        
        eval_dataset = None
        if data_config.get("validation_file"):
            eval_dataset = create_dataset(
                data_path=data_config.get("validation_file"),
                tokenizer=self.tokenizer,
                max_length=data_config.get("max_seq_length", 2048),
                mode="eval"
            )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from configuration."""
        training_config = self.config.get("training", {})
        
        # Convert config to TrainingArguments
        args = TrainingArguments(
            output_dir=training_config.get("output_dir", "./output"),
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            
            # Optimization
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            
            # Precision
            bf16=training_config.get("bf16", False),
            fp16=training_config.get("fp16", False),
            
            # Checkpointing and evaluation
            save_steps=training_config.get("save_steps", 500),
            eval_steps=training_config.get("eval_steps", 500),
            logging_steps=training_config.get("logging_steps", 10),
            save_total_limit=training_config.get("save_total_limit", 3),
            
            # Misc
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            remove_unused_columns=training_config.get("remove_unused_columns", False),
            report_to=training_config.get("report_to", []),
            
            # Distributed training
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            
            # Memory optimization
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        )
        
        return args
    
    def create_trainer(self, training_args: TrainingArguments, data_module: Dict[str, Any]) -> Trainer:
        """Create the Trainer instance."""
        from training.trainer import QwenTrainer
        
        # Create custom trainer with loss function integration
        trainer = QwenTrainer(
            model=self.model,
            args=training_args,
            train_dataset=data_module["train_dataset"],
            eval_dataset=data_module["eval_dataset"],
            tokenizer=self.tokenizer,
            # Custom loss function will be integrated in QwenTrainer
            custom_loss_fn=self.loss_fn
        )
        
        return trainer
    
    def train(self):
        """Run the training process."""
        logger.info("Starting training...")
        
        # Setup components
        self.setup_distributed()
        self.setup_model_and_tokenizer()
        self.setup_loss_function()
        
        # Create data and training components
        data_module = self.create_data_module()
        training_args = self.create_training_arguments()
        self.trainer = self.create_trainer(training_args, data_module)
        
        # Check for checkpoints
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir):
            checkpoint = get_last_checkpoint(training_args.output_dir)
        
        if checkpoint:
            logger.info(f"Resuming training from {checkpoint}")
        
        # Train
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model
        if self.dist_manager is None or self.dist_manager.should_save():
            self.trainer.save_model()
            self.trainer.save_state()
            
            # Save token configuration if special tokens were used
            if self.token_manager:
                token_config_path = os.path.join(training_args.output_dir, "token_config.yaml")
                self.token_manager.export_config(token_config_path)
                logger.info(f"Saved token configuration to {token_config_path}")
        
        # Log training results
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training completed successfully!")
        
        return train_result
    
    def evaluate(self):
        """Run evaluation."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        
        logger.info("Starting evaluation...")
        
        eval_result = self.trainer.evaluate()
        
        # Log evaluation results
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)
        
        logger.info("Evaluation completed successfully!")
        
        return eval_result
    
    def cleanup(self):
        """Cleanup resources."""
        if self.dist_manager:
            self.dist_manager.cleanup()

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5 models with custom architectures and loss functions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Prepare overrides
    overrides = {}
    if args.output_dir:
        overrides["training"] = {"output_dir": args.output_dir}
    if args.resume_from_checkpoint:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    
    # Create trainer
    trainer = Qwen25Trainer(args.config, **overrides)
    
    try:
        if args.eval_only:
            # Load model and run evaluation
            trainer.setup_distributed()
            trainer.setup_model_and_tokenizer()
            trainer.setup_loss_function()
            
            data_module = trainer.create_data_module()
            training_args = trainer.create_training_arguments()
            trainer.trainer = trainer.create_trainer(training_args, data_module)
            
            trainer.evaluate()
        else:
            # Run training
            trainer.train()
            
            # Run evaluation if eval dataset is available
            if trainer.config.get("data", {}).get("validation_file"):
                trainer.evaluate()
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()