import torch
import torch.nn as nn
import torch.nn.functional as F


from loss.triplet_loss import TripletLoss
from loss.ExponentialLogarithmicLoss import ExponentialLogarithmicLoss


class LossManager(nn.Module):
    def __init__(self, config, local=False, device="cpu"):
        super(LossManager, self).__init__()
        self.config = config
        self.device = device
        self.local = local
        self.cur_epoch = 0

        # Parse the configuration strings for re-id and semantic segmentation losses
        reid_losses = config["LOSS"].get("REID", "")
        seg_losses = config["LOSS"].get("SEG", "")

        # Loss function handles
        self.ce_loss_fn = None
        self.triplet_loss_fn = None
        self.seg_loss_fn = None
        self.direction_loss_fn = None

        self.log_softmax = None

        # Initialize loss weights and loss functions for re-id
        if "crossentropyloss" in reid_losses.lower():
            self.weight_celoss = self._get_weight(reid_losses, "CrossEntropyLoss")
            self.weight_celoss *= float(config["LOSS"]["WEIGHT_CE"])

            epsilon = config.get("LOSS", {}).get("LS_EPSILON", 0.0)
            self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=epsilon)
            self.local_ce_loss_fn = (
                nn.CrossEntropyLoss(label_smoothing=epsilon, reduction="none")
                if local
                else None
            )

        if "tripletloss" in reid_losses.lower():
            self.weight_triplet = self._get_weight(reid_losses, "TripletLoss")
            self.weight_triplet *= float(config["LOSS"]["WEIGHT_TRIPLET"])
            self.triplet_loss_fn = TripletLoss(margin=config["LOSS"]["MARGIN"])

        # Initialize loss function for semantic segmentation
        if "mseloss" in seg_losses.lower():
            self.seg_loss_fn = nn.MSELoss(reduction="mean")

        elif "kldivloss" in seg_losses.lower():
            self.log_softmax = nn.LogSoftmax(dim=1)
            self.seg_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)

        elif "bceloss" in seg_losses.lower():
            self.seg_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        elif "exponentiallogarithmicloss" in seg_losses.lower():
            w_dice = config["LOSS"].get("WEIGHT_DICE", 0.5)
            gamma = config["LOSS"].get("GAMMA", 1.0)
            self.seg_loss_fn = ExponentialLogarithmicLoss(
                w_dice=w_dice, w_cross=(1.0 - w_dice), gamma=gamma, use_softmax=True, filter=False
            )

        self.direction_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # Global loss weights from configuration
        self.weight_reid = config["LOSS"].get("WEIGHT_REID", 1.0)
        self.weight_reid = float(self.weight_reid)

        self.weight_seg = config["LOSS"].get("WEIGHT_SEG", 1.0)
        self.weight_seg = float(self.weight_seg)

        self.weight_direction = config["LOSS"].get("WEIGHT_DIRECTION", 1.0)
        self.weight_direction = float(self.weight_direction)

        # Initialize epoch tracking variables
        self.reset_epoch_loss()


    @staticmethod
    def _get_weight(losses_str: str, tgt_loss: str) -> float:
        """
        Parses a loss configuration string (e.g. "0.5*CrossEntropyLoss+TripletLoss")
        and returns the weight for a target loss type.
        """
        losses = losses_str.split("+")
        for loss in losses:
            if tgt_loss in loss:
                if "*" in loss:
                    weight = float(loss.split("*")[0])
                else:
                    weight = 1.0
                return weight
        return 0.0

    def forward(
        self,
        target,
        logits,
        feats,
        masks=None,
        gt_masks=None,
        directions=None,
        gt_directions=None,
        visibility=None,
        local_feats=None,
        local_logits=None,
        local_directions=None,
    ):
        """
        Computes the total loss from the provided inputs and logs the component losses.
        Also updates running loss statistics for the current epoch.
        """
        # Move inputs to the proper device
        target = target.to(self.device)
        logits = logits.to(self.device)

        if masks is not None:
            masks = masks.to(self.device)
        if gt_masks is not None:
            gt_masks = gt_masks.to(self.device)
        if gt_directions is not None:
            gt_directions = gt_directions.to(self.device)
        if directions is not None:
            directions = directions.to(self.device)
        if local_logits is not None:
            local_logits = local_logits.to(self.device)
        if local_directions is not None:
            local_directions = local_directions.to(self.device)

        if visibility is not None:
            visibility = visibility.to(self.device)

        reid_loss = torch.tensor(0.0).to(self.device)
        seg_loss = torch.tensor(0.0).to(self.device)

        direction_loss = torch.tensor(0.0).to(self.device)
        local_direction_loss = torch.tensor(0.0).to(self.device)

        ce_loss_val = torch.tensor(0.0).to(self.device)
        local_ce_loss_val = torch.tensor(0.0).to(self.device)

        triplet_loss_val = torch.tensor(0.0).to(self.device)

        # Compute Cross Entropy loss if available
        if self.ce_loss_fn is not None:
            if logits is not None:
                ce_loss_val += self.ce_loss_fn(logits, target)

            if local_logits is not None:
                B, P, C = local_logits.shape

                local_logits = local_logits.view(B * P, C)
                local_target = target.unsqueeze(1).expand(B, P).reshape(-1)

                if visibility is not None:
                    vis_mask = visibility.view(-1).bool()
                    ce_sum = self.local_ce_loss_fn(
                        local_logits[vis_mask], local_target[vis_mask]
                    ).sum()
                    vis_parts = vis_mask.sum().item()

                else:
                    ce_sum = self.local_ce_loss_fn(local_logits, local_target).sum()
                    vis_parts = B * P

                if vis_parts > 0:
                    local_ce_loss_val = ce_sum / vis_parts

            if local_ce_loss_val > 0.0:
                ce_loss_val += local_ce_loss_val
                ce_loss_val /= 2.0  # Average the local and global CE losses

            reid_loss += self.weight_celoss * ce_loss_val

        # Compute Triplet loss if available
        if self.triplet_loss_fn is not None:
            triplet_loss_val = self.triplet_loss_fn(
                global_feat=feats,
                labels=target,
                local_feats=local_feats,
                visibility=visibility,
                dist="cosine",
            )
            reid_loss += self.weight_triplet * triplet_loss_val

        # Compute segmentation loss if available
        if self.seg_loss_fn is not None:
            gt_masks = gt_masks.float()

            if self.log_softmax is not None:
                masks = self.log_softmax(masks)
                # Passing softmax to the ground truth masks
                gt_masks = F.softmax(gt_masks, dim=1)
                gt_masks = nn.functional.softmax(gt_masks, dim=1)

            seg_loss = self.seg_loss_fn(masks, gt_masks)

        # Compute direction loss if available
        if self.direction_loss_fn is not None:
            if directions is not None:
                direction_loss = self.direction_loss_fn(
                    directions, gt_directions.float()
                ).mean()

            if local_directions is not None:
                B, P, C = local_directions.shape

                logits_flat = local_directions.view(B * P, C)
                targets_flat = gt_directions.repeat_interleave(P).float()

                logits_flat = logits_flat.view(-1, 1)
                targets_flat = targets_flat.view(-1, 1)

                per_part_loss = self.direction_loss_fn(logits_flat, targets_flat)
                dir_loss_matrix = per_part_loss.view(B, P)

                if visibility is not None:
                    vis_mask = visibility.bool()
                    dir_loss_matrix = dir_loss_matrix * vis_mask
                    vis_counts = vis_mask.sum(dim=0)

                    valid = vis_counts > 0

                    dir_loss_matrix = dir_loss_matrix.sum(dim=0) / vis_counts.clamp(
                        min=1
                    )

                    if valid.any():
                        local_direction_loss = dir_loss_matrix[valid].mean()

                else:
                    local_direction_loss = dir_loss_matrix.mean(dim=0)

            if local_direction_loss > 0.0 and direction_loss > 0.0:
                direction_loss += local_direction_loss
                direction_loss /= 2.0
            else:
                direction_loss += local_direction_loss

        total_loss = (
            reid_loss * self.weight_reid
            + seg_loss * self.weight_seg
            + direction_loss * self.weight_direction
        )

        # Update epoch statistics
        self.epoch_total_loss += total_loss.item()
        self.epoch_reid_loss += (reid_loss).item()
        self.epoch_triplet_loss += triplet_loss_val
        self.epoch_ce_loss += (ce_loss_val).item()
        self.epoch_seg_loss += (seg_loss).item()
        self.epoch_direction_loss += (direction_loss).item()
        self.epoch_batches += 1

        return total_loss

    def get_epoch_loss(self):
        """
        Returns the average total, re-id, and semantic segmentation loss values for the current epoch,
        then resets the running totals.
        """
        if self.epoch_batches == 0:
            return {"total_loss": 0.0, "reid_loss": 0.0, "seg_loss": 0.0}

        avg_total = self.epoch_total_loss / self.epoch_batches
        avg_reid = self.epoch_reid_loss / self.epoch_batches
        avg_triplet = self.epoch_triplet_loss / self.epoch_batches
        avg_ce = self.epoch_ce_loss / self.epoch_batches
        avg_seg = self.epoch_seg_loss / self.epoch_batches
        avg_direction = self.epoch_direction_loss / self.epoch_batches

        stats = {
            "total_loss": avg_total,
            "reid_loss": avg_reid,
            "triplet_loss": avg_triplet,
            "ce_loss": avg_ce,
            "seg_loss": avg_seg,
            "direction_loss": avg_direction,
        }
        self.reset_epoch_loss()
        return stats

    def reset_epoch_loss(self):
        """
        Resets the accumulated loss statistics for a new epoch.
        """
        self.epoch_total_loss = 0.0
        self.epoch_reid_loss = 0.0
        self.epoch_triplet_loss = 0.0
        self.epoch_ce_loss = 0.0
        self.epoch_seg_loss = 0.0
        self.epoch_direction_loss = 0.0
        self.epoch_batches = 0
