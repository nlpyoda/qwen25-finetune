"""
Distributed training utilities and configuration.

This module provides easy-to-use utilities for setting up and managing
distributed training across multiple nodes and GPUs. It handles NCCL
configuration, environment setup, and provides high-level abstractions
for common distributed training patterns.
"""

import os
import socket
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.
    
    This class centralizes all distributed training configuration
    to make it easy to manage and modify distributed training setups.
    """
    # Basic distributed settings
    backend: str = "nccl"
    nodes: int = 1
    node_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    
    # Network configuration
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # NCCL optimization settings
    nccl_timeout_minutes: int = 30
    nccl_debug: str = "WARN"
    nccl_socket_ifname: str = "^docker0,lo"
    nccl_ib_disable: bool = False
    nccl_tree_threshold: int = 0
    nccl_algo: str = "Tree"
    
    # Performance settings
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # Multi-node specific settings
    use_infiniband: bool = True
    optimize_for_aws: bool = True
    instance_type: str = "p4d.24xlarge"
    
    # Environment settings
    cuda_device_order: str = "PCI_BUS_ID"
    pytorch_cuda_alloc_conf: str = "expandable_segments:True"
    omp_num_threads: int = 16
    mkl_num_threads: int = 16

class DistributedManager:
    """
    Manager for distributed training setup and coordination.
    
    This class provides a high-level interface for setting up distributed
    training, handling environment configuration, and coordinating between
    processes.
    
    Example:
        ```python
        # Initialize distributed training
        dist_manager = DistributedManager(config)
        dist_manager.setup()
        
        # Wrap model for distributed training
        model = dist_manager.wrap_model(model)
        
        # Training loop with distributed utilities
        for batch in dataloader:
            if dist_manager.should_log():
                logger.info("Training step...")
            
            loss = model(batch)
            dist_manager.backward(loss)
            dist_manager.step()
        
        # Cleanup
        dist_manager.cleanup()
        ```
    """
    
    def __init__(self, config: DistributedConfig):
        """
        Initialize distributed manager.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.is_initialized = False
        self.is_main_process = False
        self.local_rank = config.local_rank
        self.world_size = config.world_size
        self.rank = config.node_rank * torch.cuda.device_count() + config.local_rank
        
        # Setup environment variables
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables for optimal distributed training."""
        env_vars = {
            # Basic distributed settings
            "MASTER_ADDR": self.config.master_addr,
            "MASTER_PORT": str(self.config.master_port),
            "WORLD_SIZE": str(self.config.world_size),
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.config.local_rank),
            
            # NCCL settings
            "NCCL_TIMEOUT": str(self.config.nccl_timeout_minutes * 60),
            "NCCL_DEBUG": self.config.nccl_debug,
            "NCCL_SOCKET_IFNAME": self.config.nccl_socket_ifname,
            "NCCL_TREE_THRESHOLD": str(self.config.nccl_tree_threshold),
            "NCCL_ALGO": self.config.nccl_algo,
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            
            # CUDA settings
            "CUDA_DEVICE_ORDER": self.config.cuda_device_order,
            "PYTORCH_CUDA_ALLOC_CONF": self.config.pytorch_cuda_alloc_conf,
            
            # CPU settings
            "OMP_NUM_THREADS": str(self.config.omp_num_threads),
            "MKL_NUM_THREADS": str(self.config.mkl_num_threads),
            "TOKENIZERS_PARALLELISM": "false"
        }
        
        # AWS/InfiniBand specific settings
        if self.config.optimize_for_aws and self.config.use_infiniband:
            env_vars.update({
                "NCCL_IB_DISABLE": "0" if self.config.use_infiniband else "1",
                "NCCL_IB_GID_INDEX": "3",
                "NCCL_IB_TC": "136",
                "NCCL_IB_TIMEOUT": "14",
                "NCCL_IB_RETRY_CNT": "7",
                "NCCL_BUFFSIZE": "8388608",
                "NCCL_NTHREADS": "16"
            })
        
        # p4d instance specific optimizations
        if "p4d" in self.config.instance_type:
            env_vars.update({
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # p4d has 8 A100s
                "NCCL_TOPO_FILE": "/opt/amazon/efa/share/hwloc/xml/p4d.24xlarge.xml"
            })
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def setup(self) -> bool:
        """
        Initialize distributed training.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.config.backend = "gloo"
            
            # Set CUDA device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=torch.distributed.default_pg_timeout
                )
            
            self.is_initialized = True
            self.is_main_process = (self.rank == 0)
            
            # Synchronize all processes
            if dist.is_initialized():
                dist.barrier()
            
            if self.is_main_process:
                print(f"Distributed training initialized:")
                print(f"  Backend: {self.config.backend}")
                print(f"  World size: {self.world_size}")
                print(f"  Nodes: {self.config.nodes}")
                print(f"  Local rank: {self.local_rank}")
                print(f"  Global rank: {self.rank}")
                print(f"  CUDA devices: {torch.cuda.device_count()}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            return False
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model (DDP or original if not distributed)
        """
        if not self.is_initialized or self.world_size == 1:
            return model
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.to(f"cuda:{self.local_rank}")
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        return ddp_model
    
    def should_log(self) -> bool:
        """Check if current process should log (only main process)."""
        return self.is_main_process
    
    def should_save(self) -> bool:
        """Check if current process should save checkpoints (only main process)."""
        return self.is_main_process
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        Perform all-reduce operation on tensor.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if not self.is_initialized:
            return tensor
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of tensors from all processes
        """
        if not self.is_initialized:
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def cleanup(self):
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            
            if self.is_main_process:
                print("Distributed training cleaned up")

class DistributedConfigBuilder:
    """
    Builder for creating distributed training configurations.
    
    This class provides a fluent interface for building distributed
    configurations for different scenarios (single-node, multi-node, etc.).
    
    Example:
        ```python
        # Multi-node configuration
        config = (DistributedConfigBuilder()
                 .for_multi_node(nodes=2, node_rank=0)
                 .with_master("10.0.1.100", 29500)
                 .optimize_for_aws()
                 .with_p4d_settings()
                 .build())
        ```
    """
    
    def __init__(self):
        self._config = DistributedConfig()
    
    def for_single_node(self, gpus: int = 8) -> 'DistributedConfigBuilder':
        """Configure for single-node training."""
        self._config.nodes = 1
        self._config.node_rank = 0
        self._config.world_size = gpus
        return self
    
    def for_multi_node(self, nodes: int, node_rank: int, gpus_per_node: int = 8) -> 'DistributedConfigBuilder':
        """Configure for multi-node training."""
        self._config.nodes = nodes
        self._config.node_rank = node_rank
        self._config.world_size = nodes * gpus_per_node
        return self
    
    def with_master(self, addr: str, port: int = 29500) -> 'DistributedConfigBuilder':
        """Set master node address and port."""
        self._config.master_addr = addr
        self._config.master_port = port
        return self
    
    def with_backend(self, backend: str) -> 'DistributedConfigBuilder':
        """Set distributed backend."""
        self._config.backend = backend
        return self
    
    def with_nccl_settings(
        self,
        timeout_minutes: int = 30,
        debug: str = "WARN",
        algo: str = "Tree"
    ) -> 'DistributedConfigBuilder':
        """Configure NCCL settings."""
        self._config.nccl_timeout_minutes = timeout_minutes
        self._config.nccl_debug = debug
        self._config.nccl_algo = algo
        return self
    
    def optimize_for_aws(self) -> 'DistributedConfigBuilder':
        """Apply AWS-specific optimizations."""
        self._config.optimize_for_aws = True
        self._config.use_infiniband = True
        return self
    
    def with_p4d_settings(self) -> 'DistributedConfigBuilder':
        """Configure for p4d instances."""
        self._config.instance_type = "p4d.24xlarge"
        return self
    
    def with_performance_settings(
        self,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False
    ) -> 'DistributedConfigBuilder':
        """Configure performance settings."""
        self._config.find_unused_parameters = find_unused_parameters
        self._config.gradient_as_bucket_view = gradient_as_bucket_view
        self._config.static_graph = static_graph
        return self
    
    def from_yaml(self, config_path: str) -> 'DistributedConfigBuilder':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        distributed_config = config_dict.get("distributed", {})
        
        for key, value in distributed_config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        return self
    
    def from_env(self) -> 'DistributedConfigBuilder':
        """Load configuration from environment variables."""
        env_mappings = {
            "WORLD_SIZE": ("world_size", int),
            "RANK": ("node_rank", int),  # Assuming RANK refers to node rank
            "LOCAL_RANK": ("local_rank", int),
            "MASTER_ADDR": ("master_addr", str),
            "MASTER_PORT": ("master_port", int)
        }
        
        for env_var, (attr_name, type_func) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                setattr(self._config, attr_name, type_func(value))
        
        return self
    
    def build(self) -> DistributedConfig:
        """Build the final configuration."""
        # Auto-detect local rank if not set
        if self._config.local_rank == 0 and "LOCAL_RANK" in os.environ:
            self._config.local_rank = int(os.environ["LOCAL_RANK"])
        
        return self._config

def create_distributed_config_from_yaml(config_path: str) -> DistributedConfig:
    """Convenience function to create config from YAML."""
    return DistributedConfigBuilder().from_yaml(config_path).build()

def create_distributed_config_from_env() -> DistributedConfig:
    """Convenience function to create config from environment."""
    return DistributedConfigBuilder().from_env().build()

def auto_detect_distributed_config() -> DistributedConfig:
    """Auto-detect distributed configuration from environment."""
    builder = DistributedConfigBuilder().from_env()
    
    # Detect if running on AWS
    try:
        # Simple heuristic: check for EFA network interface
        result = subprocess.run(["ip", "link", "show"], capture_output=True, text=True)
        if "efa" in result.stdout:
            builder.optimize_for_aws()
    except:
        pass
    
    return builder.build()

class DistributedDataLoader:
    """
    Wrapper for creating distributed-aware data loaders.
    
    This class handles the creation of distributed samplers and
    ensures proper data distribution across processes.
    """
    
    def __init__(self, dist_manager: DistributedManager):
        """
        Initialize distributed data loader wrapper.
        
        Args:
            dist_manager: Distributed manager instance
        """
        self.dist_manager = dist_manager
    
    def create_sampler(self, dataset, shuffle: bool = True):
        """Create distributed sampler for dataset."""
        if not self.dist_manager.is_initialized:
            return None
        
        from torch.utils.data.distributed import DistributedSampler
        
        return DistributedSampler(
            dataset,
            num_replicas=self.dist_manager.world_size,
            rank=self.dist_manager.rank,
            shuffle=shuffle
        )
    
    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ):
        """Create distributed data loader."""
        from torch.utils.data import DataLoader
        
        sampler = self.create_sampler(dataset, shuffle)
        
        # Don't shuffle if using distributed sampler
        if sampler is not None:
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs
        )

def save_distributed_config(config: DistributedConfig, output_path: str):
    """Save distributed configuration to YAML file."""
    config_dict = {
        "distributed": {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def test_distributed_connectivity(config: DistributedConfig) -> bool:
    """
    Test distributed connectivity between nodes.
    
    Args:
        config: Distributed configuration
        
    Returns:
        True if connectivity test passes, False otherwise
    """
    try:
        dist_manager = DistributedManager(config)
        success = dist_manager.setup()
        
        if success:
            # Simple connectivity test
            test_tensor = torch.ones(1) * config.node_rank
            if torch.cuda.is_available():
                test_tensor = test_tensor.cuda()
            
            # All-reduce test
            dist_manager.all_reduce(test_tensor)
            
            # Check if result is correct (sum of all ranks)
            expected_sum = sum(range(config.nodes))
            if abs(test_tensor.item() - expected_sum) < 1e-6:
                if dist_manager.is_main_process:
                    print("✅ Distributed connectivity test passed")
                return True
            else:
                if dist_manager.is_main_process:
                    print("❌ Distributed connectivity test failed")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ Distributed connectivity test failed: {e}")
        return False
    
    finally:
        if 'dist_manager' in locals():
            dist_manager.cleanup()