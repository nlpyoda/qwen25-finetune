"""
Custom loss function implementations.

This module contains various custom loss functions that can be used
for specialized training scenarios with Qwen models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .base import BaseLoss, LossOutput, MultiComponentLoss, AdaptiveLoss

class FocalLoss(BaseLoss):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss reduces the relative loss for well-classified examples
    and focuses learning on hard negatives.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        **kwargs
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Compute focal loss."""
        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Mask out ignored tokens
        mask = labels_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        labels_flat = labels_flat[mask]
        
        if logits_flat.numel() == 0:
            return LossOutput(loss=torch.tensor(0.0, device=logits.device, requires_grad=True))
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        metrics = {
            "focal_alpha": self.alpha,
            "focal_gamma": self.gamma,
            "avg_confidence": pt.mean().item(),
            "hard_examples_ratio": (pt < 0.5).float().mean().item()
        }
        
        return LossOutput(loss=loss, metrics=metrics)

class LabelSmoothingLoss(BaseLoss):
    """
    Label Smoothing Cross Entropy Loss.
    
    Applies label smoothing to prevent overconfident predictions
    and improve model calibration.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        **kwargs
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Compute label smoothing loss."""
        vocab_size = logits.size(-1)
        
        # Reshape for computation
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Mask out ignored tokens
        mask = labels_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        labels_flat = labels_flat[mask]
        
        if logits_flat.numel() == 0:
            return LossOutput(loss=torch.tensor(0.0, device=logits.device, requires_grad=True))
        
        # Apply log softmax
        log_probs = F.log_softmax(logits_flat, dim=-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        smooth_targets.scatter_(1, labels_flat.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Compute metrics
        with torch.no_grad():
            ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
            perplexity = torch.exp(ce_loss)
        
        metrics = {
            "smoothing_factor": self.smoothing,
            "cross_entropy": ce_loss.item(),
            "perplexity": perplexity.item()
        }
        
        return LossOutput(loss=loss, metrics=metrics)

class ContrastiveLoss(BaseLoss):
    """
    Contrastive Loss for representation learning.
    
    Useful for training models to learn better representations
    by contrasting positive and negative examples.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        **kwargs
    ):
        """
        Initialize Contrastive Loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities
            margin: Margin for negative examples
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.margin = margin
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        representations: Optional[torch.Tensor] = None,
        **kwargs
    ) -> LossOutput:
        """Compute contrastive loss."""
        if representations is None:
            # If no representations provided, use standard cross entropy
            return LossOutput(
                loss=F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.ignore_index
                )
            )
        
        batch_size = representations.size(0)
        
        # Normalize representations
        representations = F.normalize(representations, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for contrastive learning
        # Assume positive pairs are adjacent in the batch
        contrastive_labels = torch.arange(batch_size, device=representations.device)
        
        # Compute contrastive loss
        contrastive_loss = F.cross_entropy(similarity_matrix, contrastive_labels)
        
        # Compute standard language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index
        )
        
        # Combine losses
        total_loss = lm_loss + self.weight * contrastive_loss
        
        metrics = {
            "contrastive_loss": contrastive_loss.item(),
            "language_modeling_loss": lm_loss.item(),
            "temperature": self.temperature,
            "avg_similarity": similarity_matrix.mean().item()
        }
        
        return LossOutput(loss=total_loss, metrics=metrics)

class KLDivergenceLoss(BaseLoss):
    """
    KL Divergence Loss for knowledge distillation.
    
    Useful for training student models to match the output
    distribution of teacher models.
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        **kwargs
    ):
        """
        Initialize KL Divergence Loss.
        
        Args:
            temperature: Temperature for softening distributions
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        **kwargs
    ) -> LossOutput:
        """Compute KL divergence loss."""
        if teacher_logits is None:
            # If no teacher logits, fall back to standard cross entropy
            return LossOutput(
                loss=F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.ignore_index
                )
            )
        
        # Reshape tensors
        student_logits = logits.view(-1, logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Mask out ignored tokens
        mask = labels_flat != self.ignore_index
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        labels_flat = labels_flat[mask]
        
        if student_logits.numel() == 0:
            return LossOutput(loss=torch.tensor(0.0, device=logits.device, requires_grad=True))
        
        # Compute soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Compute standard cross entropy for hard targets
        ce_loss = F.cross_entropy(student_logits, labels_flat, reduction='mean')
        
        # Combine losses
        total_loss = (1 - self.weight) * ce_loss + self.weight * kl_loss
        
        metrics = {
            "kl_divergence": kl_loss.item(),
            "cross_entropy": ce_loss.item(),
            "temperature": self.temperature,
            "distillation_weight": self.weight
        }
        
        return LossOutput(loss=total_loss, metrics=metrics)

class CurriculumLoss(AdaptiveLoss):
    """
    Curriculum Learning Loss that gradually increases difficulty.
    
    This loss starts with easier examples and gradually includes
    more difficult ones as training progresses.
    """
    
    def __init__(
        self,
        difficulty_schedule: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize Curriculum Loss.
        
        Args:
            difficulty_schedule: Schedule for increasing difficulty
            **kwargs: Additional arguments passed to AdaptiveLoss
        """
        super().__init__(adaptation_schedule=difficulty_schedule, **kwargs)
        self.max_sequence_length = difficulty_schedule.get("max_sequence_length", 2048)
        self.initial_length = difficulty_schedule.get("initial_length", 256)
        self.length_increment = difficulty_schedule.get("length_increment", 128)
    
    def _compute_adaptive_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> LossOutput:
        """Compute curriculum loss with adaptive sequence length."""
        # Determine current maximum sequence length
        current_max_length = min(
            self.initial_length + self.current_phase * self.length_increment,
            self.max_sequence_length
        )
        
        # Truncate sequences if necessary
        if logits.size(1) > current_max_length:
            logits = logits[:, :current_max_length, :]
            labels = labels[:, :current_max_length]
        
        # Compute standard cross entropy
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        
        metrics = {
            "curriculum_phase": self.current_phase,
            "current_max_length": current_max_length,
            "sequence_utilization": logits.size(1) / self.max_sequence_length
        }
        
        return LossOutput(loss=loss, metrics=metrics)

class MultimodalAlignmentLoss(BaseLoss):
    """
    Loss for aligning vision and text representations.
    
    This loss encourages the model to learn aligned representations
    between different modalities (vision and text).
    """
    
    def __init__(
        self,
        alignment_temperature: float = 0.07,
        **kwargs
    ):
        """
        Initialize Multimodal Alignment Loss.
        
        Args:
            alignment_temperature: Temperature for alignment loss
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.alignment_temperature = alignment_temperature
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> LossOutput:
        """Compute multimodal alignment loss."""
        # Standard language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index
        )
        
        auxiliary_losses = {}
        metrics = {"language_modeling_loss": lm_loss.item()}
        
        # Compute alignment loss if both modalities are present
        if vision_features is not None and text_features is not None:
            # Normalize features
            vision_features = F.normalize(vision_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Compute similarity
            similarity = torch.matmul(vision_features, text_features.T) / self.alignment_temperature
            
            # Create positive pair labels (assume diagonal pairing)
            batch_size = vision_features.size(0)
            labels_alignment = torch.arange(batch_size, device=vision_features.device)
            
            # Compute contrastive loss for both directions
            v2t_loss = F.cross_entropy(similarity, labels_alignment)
            t2v_loss = F.cross_entropy(similarity.T, labels_alignment)
            
            alignment_loss = (v2t_loss + t2v_loss) / 2
            auxiliary_losses["alignment_loss"] = alignment_loss
            
            metrics.update({
                "alignment_loss": alignment_loss.item(),
                "v2t_loss": v2t_loss.item(),
                "t2v_loss": t2v_loss.item(),
                "avg_similarity": similarity.diag().mean().item()
            })
        
        return LossOutput(
            loss=lm_loss,
            auxiliary_losses=auxiliary_losses if auxiliary_losses else None,
            metrics=metrics
        )

class TokenTypeLoss(BaseLoss):
    """
    Loss that applies different weights to different token types.
    
    This loss allows applying different importance weights to
    different types of tokens (e.g., special tokens, function tokens, etc.).
    """
    
    def __init__(
        self,
        token_weights: Dict[str, float],
        default_weight: float = 1.0,
        **kwargs
    ):
        """
        Initialize Token Type Loss.
        
        Args:
            token_weights: Dictionary mapping token IDs to weights
            default_weight: Default weight for tokens not in token_weights
            **kwargs: Additional arguments passed to BaseLoss
        """
        super().__init__(**kwargs)
        self.token_weights = token_weights
        self.default_weight = default_weight
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokenizer=None,
        **kwargs
    ) -> LossOutput:
        """Compute token-type-weighted loss."""
        # Reshape tensors
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Mask out ignored tokens
        mask = labels_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        labels_flat = labels_flat[mask]
        
        if logits_flat.numel() == 0:
            return LossOutput(loss=torch.tensor(0.0, device=logits.device, requires_grad=True))
        
        # Compute per-token losses
        token_losses = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        
        # Create weight tensor
        weights = torch.ones_like(labels_flat, dtype=torch.float) * self.default_weight
        
        # Apply custom weights for specific tokens
        for token_id, weight in self.token_weights.items():
            if isinstance(token_id, str) and tokenizer is not None:
                # Convert token string to ID
                token_id = tokenizer.convert_tokens_to_ids(token_id)
            
            if isinstance(token_id, int):
                token_mask = labels_flat == token_id
                weights[token_mask] = weight
        
        # Apply weights to losses
        weighted_losses = token_losses * weights
        
        if self.reduction == 'mean':
            loss = weighted_losses.mean()
        elif self.reduction == 'sum':
            loss = weighted_losses.sum()
        else:
            loss = weighted_losses
        
        # Compute metrics
        metrics = {
            "avg_token_weight": weights.mean().item(),
            "weighted_tokens_ratio": (weights != self.default_weight).float().mean().item()
        }
        
        return LossOutput(loss=loss, metrics=metrics)