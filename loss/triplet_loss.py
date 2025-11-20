import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection / (x_norm * y_norm)
    dist = (1.0 - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )

    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
    )
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        )
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data
        )
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using hard example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction="mean")
            self.parts_ranking_loss = nn.MarginRankingLoss(
                margin=margin, reduction="sum"
            )

        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction="mean")
            self.parts_ranking_loss = nn.SoftMarginLoss(reduction="sum")

    def __call__(
        self,
        global_feat,
        labels,
        local_feats=None,
        visibility=None,
        normalize_feature=False,
        dist="cosine",
    ):

        # Normalize features if requested.
        if normalize_feature:
            if global_feat is not None:
                global_feat = normalize(global_feat, axis=-1)
            if local_feats is not None:
                local_feats = normalize(local_feats, axis=-1)

        # Choose the appropriate distance function.
        dist_func = euclidean_dist
        if dist == "cosine":
            dist_func = cosine_dist

        global_loss = torch.tensor(0.0, device=labels.device)
        local_loss = torch.tensor(0.0, device=global_feat.device)

        # Compute global loss only if global_feat is provided.
        if global_feat is not None:
            global_feat = global_feat.squeeze()
            global_loss += self._get_loss(
                feats=global_feat,
                dist_func=dist_func,
                labels=labels,
                loss_func=self.ranking_loss,
                margin=self.margin,
            )

        # Compute local loss if local_feats is provided.
        if local_feats is not None:
            num_samples = local_feats.size(0)
            num_parts = local_feats.size(1)

            valid_parts_count = 0

            for i in range(num_parts):
                part_feats = local_feats[:, i, :]
                part_visibility = visibility[:, i]

                part_loss = self._get_loss(
                    feats=part_feats,
                    dist_func=dist_func,
                    labels=labels,
                    loss_func=self.parts_ranking_loss,
                    margin=self.margin,
                    visibility=part_visibility,
                )
                num_valid_samples_per_part = 0

                if part_visibility is not None:
                    num_valid_samples_per_part = part_visibility.float().sum()
                else:
                    num_valid_samples_per_part = num_samples

                if num_valid_samples_per_part > 0:
                    mean_loss_per_part = part_loss / num_valid_samples_per_part
                    local_loss += mean_loss_per_part
                    valid_parts_count += 1

            if valid_parts_count > 0:
                # Average the local loss over the number of valid parts.
                local_loss = local_loss / valid_parts_count
            else:
                # If no valid parts were found, set local_loss to zero.
                local_loss = torch.tensor(0.0, device=local_feats.device)

        final_loss = torch.tensor(0.0, device=labels.device)
        has_global = global_feat is not None and global_feat.numel() > 0
        has_local = (
            local_feats is not None
            and local_feats.numel() > 0
            and valid_parts_count > 0
        )

        if has_global and has_local:
            final_loss = (global_loss + local_loss) / 2.0
        elif has_global:
            final_loss = global_loss
        elif has_local:
            final_loss = local_loss

        return final_loss

    def _get_loss(
        self, feats, dist_func, labels, loss_func, margin=None, visibility=None
    ):
        dist_mat = dist_func(feats, feats)

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        dist_ap *= 1.0 + self.hard_factor
        dist_an *= 1.0 - self.hard_factor

        if visibility is not None:
            visible_mask = visibility.bool()

            dist_ap_filtered = dist_ap[visible_mask]
            dist_an_filtered = dist_an[visible_mask]

        else:
            dist_ap_filtered = dist_ap
            dist_an_filtered = dist_an

        if dist_ap_filtered.numel() == 0 or dist_an_filtered.numel() == 0:
            return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

        y = dist_an_filtered.new_ones(dist_an_filtered.size())

        if margin is not None:
            loss = loss_func(dist_an_filtered, dist_ap_filtered, y)

        else:
            input = dist_an_filtered - dist_ap_filtered
            loss = loss_func(input, y)

        return loss
