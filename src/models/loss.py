import torch
import torch.nn as nn
import torch.nn.functional as F

def yoho_detection_loss(preds, targets, lambda_coord=5.0):
    """
    preds: [B, N, 2 + C] -> [start, end, class_logits]
    targets: list of [N, 3] per sample -> [start, end, class_id], normalized to [0, 1]

    Returns scalar loss
    """
    B, N, D = preds.shape
    num_classes = D - 2
    device = preds.device

    total_loc_loss = 0.0
    total_cls_loss = 0.0

    for b in range(B):
        pred = preds[b]  # [N, 2 + C]
        target = targets[b]  # [M, 3] (variable number of GT events)

        # Assign each target to the closest proposal cell (e.g., based on IoU or center)
        # Simplified: nearest start bin
        assigned = torch.zeros(N, dtype=torch.bool, device=device)

        for t in target:
            s_gt, e_gt, c_gt = t

            # Find best matching cell (e.g., mid-point bin)
            mid_gt = 0.5 * (s_gt + e_gt)
            i = int(mid_gt * N)  # assumes N grid cells uniformly across 5s
            i = min(max(i, 0), N - 1)  # clamp

            if assigned[i]: continue  # skip if already matched
            assigned[i] = True

            s_pred, e_pred = pred[i, 0], pred[i, 1]
            cls_logits = pred[i, 2:]

            # Localization loss: SmoothL1 between start/end
            loc_loss = F.smooth_l1_loss(s_pred, s_gt) + F.smooth_l1_loss(e_pred, e_gt)

            # Classification loss
            cls_loss = F.cross_entropy(cls_logits.unsqueeze(0), c_gt.long().unsqueeze(0))

            total_loc_loss += loc_loss
            total_cls_loss += cls_loss

    total_loss = lambda_coord * total_loc_loss + total_cls_loss
    return total_loss / B
