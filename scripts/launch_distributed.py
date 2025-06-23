#!/usr/bin/env python3
"""
Launcher script for distributed training with automatic configuration.

This script provides a high-level interface for launching distributed
training jobs with automatic environment detection and configuration.
"""

import argparse
import os
import sys
import subprocess
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.distributed import (
    DistributedConfig, DistributedConfigBuilder, 
    auto_detect_distributed_config, save_distributed_config
)

class DistributedLauncher:
    """
    High-level launcher for distributed training jobs.
    
    This class handles the complexity of launching distributed training
    across different environments (single-node, multi-node, different
    cloud providers, etc.).
    """
    
    def __init__(self):
        self.config: Optional[DistributedConfig] = None
        self.training_script = "train.py"
        self.script_args: List[str] = []
    
    def auto_configure(self) -> DistributedConfig:
        """Auto-detect and configure distributed settings."""
        print("🔍 Auto-detecting distributed configuration...")
        
        config = auto_detect_distributed_config()
        
        # Enhanced detection for specific environments
        config = self._detect_environment_specifics(config)
        
        print(f"✅ Detected configuration:")
        print(f"   Backend: {config.backend}")
        print(f"   Nodes: {config.nodes}")
        print(f"   World size: {config.world_size}")
        print(f"   Instance type: {config.instance_type}")
        
        return config
    
    def _detect_environment_specifics(self, config: DistributedConfig) -> DistributedConfig:
        """Detect environment-specific settings."""
        
        # Detect AWS instance type
        try:
            import requests
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type",
                timeout=2
            )
            if response.status_code == 200:
                instance_type = response.text
                config.instance_type = instance_type
                
                if "p4d" in instance_type:
                    config.optimize_for_aws = True
                    config.use_infiniband = True
                    print(f"🚀 Detected AWS {instance_type} - applying optimizations")
                
        except:
            # Not running on AWS or metadata service unavailable
            pass
        
        # Detect SLURM environment
        if "SLURM_JOB_ID" in os.environ:
            print("🔧 Detected SLURM environment")
            config = self._configure_for_slurm(config)
        
        return config
    
    def _configure_for_slurm(self, config: DistributedConfig) -> DistributedConfig:
        """Configure for SLURM cluster environment."""
        
        # Get SLURM environment variables
        job_id = os.environ.get("SLURM_JOB_ID")
        node_id = int(os.environ.get("SLURM_PROCID", "0"))
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        
        # Get node list and determine master
        node_list = os.environ.get("SLURM_JOB_NODELIST", "localhost")
        
        # Simple master node selection (first node)
        master_node = node_list.split(",")[0] if "," in node_list else node_list
        
        config.nodes = num_nodes
        config.node_rank = node_id
        config.master_addr = master_node
        
        print(f"📋 SLURM configuration:")
        print(f"   Job ID: {job_id}")
        print(f"   Node rank: {node_id}/{num_nodes}")
        print(f"   Master: {master_node}")
        
        return config
    
    def configure_from_file(self, config_path: str) -> DistributedConfig:
        """Load configuration from YAML file."""
        print(f"📄 Loading configuration from {config_path}")
        
        builder = DistributedConfigBuilder().from_yaml(config_path)
        
        # Override with environment variables if present
        builder.from_env()
        
        return builder.build()
    
    def configure_manual(
        self,
        nodes: int,
        node_rank: int,
        master_addr: str,
        master_port: int = 29500,
        gpus_per_node: int = 8
    ) -> DistributedConfig:
        """Manual configuration for distributed training."""
        print(f"⚙️  Manual configuration:")
        print(f"   Nodes: {nodes}")
        print(f"   Node rank: {node_rank}")
        print(f"   Master: {master_addr}:{master_port}")
        
        return (DistributedConfigBuilder()
                .for_multi_node(nodes, node_rank, gpus_per_node)
                .with_master(master_addr, master_port)
                .optimize_for_aws()  # Default to AWS optimizations
                .build())
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        if not self.config:
            print("❌ No configuration loaded")
            return False
        
        print("🔍 Validating configuration...")
        
        # Check basic requirements
        if self.config.nodes < 1:
            print("❌ Invalid number of nodes")
            return False
        
        if self.config.world_size < 1:
            print("❌ Invalid world size")
            return False
        
        # Check network connectivity for multi-node
        if self.config.nodes > 1:
            if not self._test_network_connectivity():
                print("❌ Network connectivity test failed")
                return False
        
        # Check CUDA availability
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.config.backend = "gloo"
        else:
            available_gpus = torch.cuda.device_count()
            expected_gpus = self.config.world_size // self.config.nodes
            if available_gpus < expected_gpus:
                print(f"⚠️  Expected {expected_gpus} GPUs, found {available_gpus}")
        
        print("✅ Configuration validation passed")
        return True
    
    def _test_network_connectivity(self) -> bool:
        """Test network connectivity to master node."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.config.master_addr, self.config.master_port))
            sock.close()
            
            if result == 0:
                print(f"✅ Network connectivity to {self.config.master_addr}:{self.config.master_port} OK")
                return True
            else:
                print(f"❌ Cannot connect to {self.config.master_addr}:{self.config.master_port}")
                return False
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            return False
    
    def generate_launch_command(self) -> List[str]:
        """Generate the torchrun launch command."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        # Base torchrun command
        cmd = ["torchrun"]
        
        # Add distributed arguments
        if self.config.nodes > 1:
            cmd.extend([
                f"--nnodes={self.config.nodes}",
                f"--node_rank={self.config.node_rank}",
                f"--master_addr={self.config.master_addr}",
                f"--master_port={self.config.master_port}"
            ])
        
        # Add GPU configuration
        gpus_per_node = self.config.world_size // self.config.nodes
        cmd.append(f"--nproc_per_node={gpus_per_node}")
        
        # Add training script and arguments
        cmd.append(self.training_script)
        cmd.extend(self.script_args)
        
        return cmd
    
    def launch(self, dry_run: bool = False) -> bool:
        """Launch the distributed training job."""
        if not self.validate_configuration():
            return False
        
        # Generate launch command
        cmd = self.generate_launch_command()
        
        print("🚀 Launching distributed training...")
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("🔍 Dry run mode - command not executed")
            return True
        
        # Setup environment variables
        self._setup_environment()
        
        # Execute training
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("⚠️  Training interrupted by user")
            return False
    
    def _setup_environment(self):
        """Setup environment variables for training."""
        # The DistributedManager will handle most environment setup,
        # but we can set some additional variables here
        
        # Set OMP and MKL threads based on configuration
        os.environ["OMP_NUM_THREADS"] = str(self.config.omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.mkl_num_threads)
        
        # Disable tokenizers parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def save_config(self, output_path: str):
        """Save the current configuration to file."""
        if self.config:
            save_distributed_config(self.config, output_path)
            print(f"💾 Configuration saved to {output_path}")

def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="Launch distributed training with automatic configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect configuration
  python launch_distributed.py --auto train.py --config my_config.yaml
  
  # Manual multi-node setup
  python launch_distributed.py --nodes 2 --node-rank 0 --master-addr 10.0.1.100 \\
                                train.py --config my_config.yaml
  
  # Load from configuration file
  python launch_distributed.py --config-file distributed_config.yaml \\
                                train.py --model qwen25_7b
  
  # Dry run to see the command
  python launch_distributed.py --auto --dry-run train.py --config my_config.yaml
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--auto", action="store_true",
        help="Auto-detect distributed configuration"
    )
    config_group.add_argument(
        "--config-file", type=str,
        help="Load configuration from YAML file"
    )
    config_group.add_argument(
        "--manual", action="store_true",
        help="Use manual configuration (requires --nodes, --node-rank, --master-addr)"
    )
    
    # Manual configuration options
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--node-rank", type=int, help="Rank of current node")
    parser.add_argument("--master-addr", type=str, help="Master node address")
    parser.add_argument("--master-port", type=int, default=29500, help="Master node port")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")
    
    # Launcher options
    parser.add_argument("--dry-run", action="store_true", help="Show command without executing")
    parser.add_argument("--save-config", type=str, help="Save configuration to file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    # Training script and arguments
    parser.add_argument("script", help="Training script to run")
    parser.add_argument("script_args", nargs="*", help="Arguments for training script")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = DistributedLauncher()
    launcher.training_script = args.script
    launcher.script_args = args.script_args
    
    # Configure based on chosen method
    try:
        if args.auto:
            launcher.config = launcher.auto_configure()
        elif args.config_file:
            launcher.config = launcher.configure_from_file(args.config_file)
        elif args.manual:
            if not all([args.nodes, args.node_rank is not None, args.master_addr]):
                parser.error("Manual configuration requires --nodes, --node-rank, and --master-addr")
            launcher.config = launcher.configure_manual(
                args.nodes, args.node_rank, args.master_addr, 
                args.master_port, args.gpus_per_node
            )
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        sys.exit(1)
    
    # Save configuration if requested
    if args.save_config:
        launcher.save_config(args.save_config)
    
    # Validate configuration
    if not launcher.validate_configuration():
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Configuration validation complete")
        sys.exit(0)
    
    # Launch training
    success = launcher.launch(dry_run=args.dry_run)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()