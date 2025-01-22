import torch
import torch.nn as nn
import logging

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class KeypointsCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(KeypointsCrossEntropyLoss, self).__init__()
        self.weights = weights.view(-1, 1, 1)  # Reshape weights for broadcasting: (C, 1, 1)

    def forward(self, out, target):
        """Forward pass with broadcasting-aware operations.
        
        Args:
            out: Model output tensor (B, C, H, W)
            target: Target heatmap tensor (B, C, H, W)
        Returns:
            Weighted cross entropy loss
        """
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        out_safe = out.clamp(min=eps)
        
        # Compute log predictions
        log_pred = torch.log(out_safe)  # (B, C, H, W)
        
        # weights is now (C, 1, 1) and will broadcast correctly
        weighted_log_pred = target * log_pred * self.weights  # (B, C, H, W)
        
        # Sum over all dimensions
        loss = -torch.sum(weighted_log_pred)
        
        # Normalize by batch size
        loss = loss / out.size(0)
        
        return loss

# class KeypointsCrossEntropyLoss(nn.Module):
#     def __init__(self, weights):
#         super(KeypointsCrossEntropyLoss, self).__init__()
#         self.weights = weights

#     def forward(self, out, target):
#         """Forward pass with detailed debugging."""
#         # logging.info("[DEBUG] Inside KeypointsCrossEntropyLoss forward")
#         # logging.info(f"[DEBUG] Input tensor stats - out: shape={out.shape}, "
#         #             f"min={out.min().item():.4f}, max={out.max().item():.4f}, "
#         #             f"mean={out.mean().item():.4f}")
#         # logging.info(f"[DEBUG] Target tensor stats: shape={target.shape}, "
#         #             f"min={target.min().item():.4f}, max={target.max().item():.4f}, "
#         #             f"mean={target.mean().item():.4f}")
        
#         # Check for any invalid values in input
#         if torch.isnan(out).any() or torch.isinf(out).any():
#             #logging.error("[DEBUG] ❌ Invalid values in prediction tensor (before log)")
#             return torch.tensor(float('nan'), device=out.device)
            
#         # Add small epsilon to prevent log(0)
#         eps = 1e-8
#         out_safe = out.clamp(min=eps)
        
#         # Compute loss with more granular debugging
#         log_pred = torch.log(out_safe)
#         # logging.info(f"[DEBUG] Log predictions stats: min={log_pred.min().item():.4f}, "
#         #             f"max={log_pred.max().item():.4f}, mean={log_pred.mean().item():.4f}")
        
#         weighted_log_pred = target * log_pred * self.weights
#         # logging.info(f"[DEBUG] Weighted predictions stats: "
#         #             f"min={weighted_log_pred.min().item():.4f}, "
#         #             f"max={weighted_log_pred.max().item():.4f}, "
#         #             f"mean={weighted_log_pred.mean().item():.4f}")
        
#         loss = -torch.sum(weighted_log_pred)
#         #logging.info(f"[DEBUG] Final loss value: {loss.item():.4f}")
        
#         return loss
