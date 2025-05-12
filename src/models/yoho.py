import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DWConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class YOHO(nn.Module):
    def __init__(self, in_shape=(1, 801, 64), num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            # Input: [B, 1, 801, 64]
            ConvBlock(1, 32, kernel_size=3, stride=2, padding=1),          # 401 × 32 × 32
            DWConvBlock(32, kernel_size=3, padding=1),
            ConvBlock(32, 64, kernel_size=1),

            DWConvBlock(64, kernel_size=3, stride=2, padding=1),           # 201 × 16 × 64
            ConvBlock(64, 128, kernel_size=1),
            DWConvBlock(128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=1),

            DWConvBlock(128, kernel_size=3, stride=2, padding=1),          # 101 × 8 × 128
            ConvBlock(128, 256, kernel_size=1),
            DWConvBlock(256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),

            DWConvBlock(256, kernel_size=3, stride=2, padding=1),          # 51 × 4 × 256
            ConvBlock(256, 512, kernel_size=1),

            *[nn.Sequential(
                DWConvBlock(512, kernel_size=3, padding=1),
                ConvBlock(512, 512, kernel_size=1)
            ) for _ in range(5)],                                          # Repeat block ×5

            DWConvBlock(512, kernel_size=3, stride=2, padding=1),          # 26 × 2 × 512
            ConvBlock(512, 1024, kernel_size=1),
            DWConvBlock(1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=1),

            DWConvBlock(1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            DWConvBlock(512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            DWConvBlock(256, kernel_size=3, padding=1),
            ConvBlock(256, 128, kernel_size=1),
        )

        # Final reshape and 1D conv
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((26, 256)), 
            nn.Flatten(start_dim=2),          # [B, 128, 26] → [B, 26, 128]
            nn.Conv1d(256, 6, kernel_size=1), # Final output: [B, 6, 26]       
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), 256, 26)  # Reshape: [B, 256, 26]
        x = self.head(x)                                      # → [B, 6, 26]
        return x.permute(0, 2, 1)                              # [B, 26, 6]



import torch
import torch.nn.functional as F

def yoho_detection_loss(preds, targets, num_classes, lambda_coord=5.0):
    """
    preds: [B, N, 2 + C] → [start, end, class_logits]
    targets: list of [M_i, 3] tensors → [start, end, class_id] normalized in [0, 1]
    num_classes: number of foreground classes (excluding background)
    Returns scalar loss
    """
    B, N, D = preds.shape
    device = preds.device
    background_class_id = num_classes  # C-th index

    total_loc_loss = 0.0
    total_cls_loss = 0.0

    for b in range(B):
        pred = preds[b]  # [N, 2 + C]
        target = targets[b]  # [M, 3]

        assigned = torch.zeros(N, dtype=torch.bool, device=device)

        for t in target:
            s_gt, e_gt, c_gt = t

            mid_gt = 0.5 * (s_gt + e_gt)
            i = int(mid_gt * N)
            i = min(max(i, 0), N - 1)

            if assigned[i]:
                continue  # skip if already assigned
            assigned[i] = True

            s_pred, e_pred = pred[i, 0], pred[i, 1]
            cls_logits = pred[i, 2:]

            loc_loss = F.smooth_l1_loss(s_pred, s_gt) + F.smooth_l1_loss(e_pred, e_gt)
            cls_loss = F.cross_entropy(cls_logits.unsqueeze(0), c_gt.long().unsqueeze(0))

            total_loc_loss += loc_loss
            total_cls_loss += cls_loss

        # Handle unmatched cells → encourage background prediction
        for i in range(N):
            if not assigned[i]:
                cls_logits = pred[i, 2:]
                background_target = torch.tensor([background_class_id], device=device)
                cls_loss = F.cross_entropy(cls_logits.unsqueeze(0), background_target)
                total_cls_loss += cls_loss

    total_loss = lambda_coord * total_loc_loss + total_cls_loss
    return total_loss / B
