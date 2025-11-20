import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from eval.ATRW.atrwtool.plain import evaluate_submission_atrw


def _fliplr(img):
    """flip horizontal"""
    inv_idx = torch.arange(
        img.size(3) - 1, -1, -1, device=img.device
    ).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def _compute_cosine_distance_matrix(
    query_features: torch.Tensor, gallery_features: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the cosine distance betwen two vectors or a pair-wise distance.

    Args:
        features   (Tensor[N, D]): first set of feature vectors
        features2  (Tensor[M, D], optional): second set of feature vectors.
            If provided, returns an (NxM) distance matrix; otherwise returns
            an (NxN) matrix for all-pairs within `features`.

    Returns:
        Tensor: 1 - cosine_similarity, shape (N, M) or (N, N)
    """
    # L2‐normalize both sets
    f1 = F.normalize(query_features, p=2, dim=1)
    if gallery_features is None:
        # self‐pairwise
        cos_sim = f1 @ f1.t()
    else:
        f2 = F.normalize(gallery_features, p=2, dim=1)
        cos_sim = f1 @ f2.t()

    # cosine distance = 1 - cosine similarity
    return 1.0 - cos_sim


def compute_part_based_cosine_distance_matrix(
    query_local,
    query_visibility,
    gallery_local=None,
    gallery_visibility=None,
    parts_weight=None,
):
    """
    Compute the cosine distance matrix for each part of the body.

    Args:
        features (Tensor): A tensor of shape (N, K, D) containing the features.
        local_features (Tensor): A tensor of shape (N, K, D) containing the local features.
        visibility (Tensor): A tensor of shape (N, K) containing the visibility of each keypoint.
    Returns:
        Tensor: A tensor of shape (N, N) containing the average cosine distance between the parts.
        If a part is not visible, the distance is set to infinite.
    """
    if gallery_local is None:
        gallery_local = query_local

    if gallery_visibility is None:
        gallery_visibility = query_visibility

    Q, K = query_visibility.shape
    G = gallery_visibility.shape[0]
    device = query_local.device

    dist = torch.zeros((Q, G), device=device)
    count = torch.zeros((Q, G), device=device)

    # Handling samples that don't have any visible parts by temporarily setting them to visibility = 1
    q_vis = query_visibility.clone()
    g_vis = gallery_visibility.clone()

    q_vis_zero = q_vis.sum(dim=1) == 0
    g_vis_zero = g_vis.sum(dim=1) == 0

    q_vis[q_vis_zero] = 1
    g_vis[g_vis_zero] = 1

    Q, K = q_vis.shape
    G = g_vis.shape[0]
    device = query_local.device

    dist = torch.zeros(Q, G, device=device)
    count = torch.zeros(Q, G, device=device)

    for k in range(K):
        # features of part k
        qf = query_local[:, k, :]
        gf = gallery_local[:, k, :]

        pdist = _compute_cosine_distance_matrix(qf, gf)

        mask = q_vis[:, k].bool().unsqueeze(1) & g_vis[:, k].bool().unsqueeze(0)

        # visibility masks
        vq = query_visibility[:, k].bool().unsqueeze(1).expand(Q, G)
        vg = gallery_visibility[:, k].bool().unsqueeze(0).expand(Q, G)
        mask = vq & vg

        # weight for this part
        w = parts_weight[k] if parts_weight is not None else 1.0

        dist[mask] += w * pdist[mask]
        count[mask] += 1

    valid = count > 0
    avg_distance = dist.clone()
    avg_distance[valid] /= count[valid]

    # Set to infinity if no part is visible
    avg_distance[~valid] = float("inf")

    return avg_distance


def compute_visibility(heatmaps, threshold=0.4, bckg_axis=True):
    """
    Compute the visibility of the keypoints given the heatmaps.

    Args:
        heatmaps (Tensor): A tensor of shape (N, K, H, W) containing the heatmaps.
        threshold (float): The threshold to consider a keypoint as visible.

    Returns:
        Tensor: A tensor of shape (N, K) containing the visibility of each keypoint.
    """
    N, K, H, W = heatmaps.shape

    if bckg_axis:
        # Ignore the last dimension of axis K
        heatmaps = heatmaps[:, :-1, :, :]
        K = K - 1

    flat_heatmap = heatmaps.view(N, K, -1)
    visibility = (flat_heatmap >= threshold).any(dim=-1).float()

    return visibility


# Function used to extract the features for evaluation
def extract_eval_feats(model, tensor, concat=True, local_test=False, vis_threshold=0.4):
    if local_test:
        local_feats, heatmaps = model.infer(tensor)
        visibility = compute_visibility(heatmaps, threshold=vis_threshold)

        if concat:
            tensor_flip = _fliplr(tensor)
            local_feats_flip, heatmaps_flip = model.infer(tensor_flip)

            local_feats = torch.cat((local_feats, local_feats_flip), dim=-1)

            # Extracting the visibility from the heatmaps
            visibility_flip = compute_visibility(heatmaps_flip, threshold=vis_threshold)

            # Doing an OR operation between the visibilities
            visibility = torch.logical_or(visibility, visibility_flip)
            return local_feats, visibility

        return local_feats, visibility

    else:
        output = model.infer(tensor)

        if concat:
            tensor_flip = _fliplr(tensor)
            output_flip = model.infer(tensor_flip)
            output = torch.cat((output, output_flip), dim=-1)

        return output


def extract_feats_from_loader(
    model, dataloader, device="cpu", concat=False
):
    feats, ids = [], []
    for batch in tqdm(dataloader, desc="Extracting features"):
        imgs, img_ids = batch
        imgs = imgs.to(device)

        f = extract_eval_feats(model, imgs, concat=concat)
        feats.append(f)

        ids.extend(img_ids)

    ids = torch.tensor(ids)
    ids = ids.to(device)

    feats = torch.cat(feats, dim=0)
    feats = feats.to(device)
    feats = feats.squeeze(1)
    return feats, ids


def calculate_cmc_map(
    distance_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor = None,
    max_k: int = 50,
):
    """
    Compute CMC and mAP from a precomputed distance matrix.
    """
    dist = distance_matrix.clone()
    Q, G = dist.shape

    # self‐match masking on square matrix
    if Q == G and gallery_labels is None:
        idx = torch.arange(Q, device=dist.device)
        dist[idx, idx] = 1000000

    labels_q = query_labels
    labels_g = gallery_labels if gallery_labels is not None else query_labels

    # adjust max_k
    if G < max_k:
        max_k = G
        print(f"Reducing max_k to {G}")

    # sort ascending distances (best matches first)
    indices = torch.argsort(dist, dim=1)

    # build match matrix
    matches = (labels_g[indices] == labels_q.unsqueeze(1)).float()

    all_cmc = []
    all_ap = []
    num_valid = 0

    for i in range(Q):
        m = matches[i]
        if m.sum() == 0:
            continue
        # CMC
        cmc = m.cumsum(0)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_k])
        num_valid += 1

        # AP
        precision = m.cumsum(0) / torch.arange(1, G + 1, device=m.device)
        ap = (precision * m).sum() / m.sum()
        all_ap.append(ap)

    assert num_valid > 0, "No valid matches found for any query!"

    cmc_scores = torch.stack(all_cmc).sum(0) / num_valid
    mAP = torch.stack(all_ap).mean().item()

    return mAP, cmc_scores


def evaluate_CMC_map(
    query_features: torch.Tensor = None,
    gallery_features: torch.Tensor = None,
    query_labels: torch.Tensor = None,
    gallery_labels: torch.Tensor = None,
    max_rank: int = 50,
):
    # Passing tensors to numpy
    query_labels = query_labels.cpu().numpy()
    gallery_labels = gallery_labels.cpu().numpy()

    distmat = _compute_cosine_distance_matrix(query_features, gallery_features)
    distmat = distmat.cpu().numpy()

    # Create query and gallery camera ids as 0s and 1s
    query_cam = np.zeros_like(query_labels)
    gallery_cam = np.ones_like(gallery_labels)

    CMC, mAP = eval_func(
        distmat=distmat,
        q_pids=query_labels,
        g_pids=gallery_labels,
        q_camids=query_cam,
        g_camids=gallery_cam,
        max_rank=max_rank,
    )

    return CMC, mAP


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """

    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_atrw(
    features,
    img_ids,
    gt_labels_path,
):
    my_result = []

    dist = _compute_cosine_distance_matrix(features)

    dist = dist.cpu().numpy()

    for i in range(len(img_ids)):
        tmp = {}
        img_name = img_ids[i]

        index = np.argsort(dist[i])
        # Erase the first element which is the query itself
        index = index[1:]

        tmp["query_id"] = int(img_name)
        gallery_tmp = []

        for j in index:
            gallery_name = int(img_ids[j])
            gallery_tmp.append(gallery_name)

        tmp["ans_ids"] = gallery_tmp
        my_result.append(tmp)


    # Now we use our result to evaluate the performances on the ATRW dataset
    return evaluate_submission_atrw(gt_labels_path, my_result, "test")


def save_model(path, model, optimizer, scheduler, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )

    print(f"Model saved at {path}.")


def load_model(path, model, optimizer=None, scheduler=None, is_train=False):
    """
    Load a model (and optionally its optimizer and scheduler) from a given path,
    dropping any parameters for the fully connected (fc) layer.
    """
    state_dict = torch.load(path)

    # Check if the state dictionary contains the key 'model_state_dict'
    if "model_state_dict" in state_dict:
        model_state_dict = state_dict["model_state_dict"]
        # Filter out keys that are 'fc' or start with 'fc.'
        filtered_state_dict = {
            k: v
            for k, v in model_state_dict.items()
            if not (k == "fc" or k.startswith("fc"))
        }
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        # In case the loaded state dict is directly the model state dict
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (k == "fc" or k.startswith("fc."))
        }
        model.load_state_dict(filtered_state_dict)

    if not is_train:
        print(f"Model loaded from {path}.")
        return model

    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    scheduler.load_state_dict(state_dict["scheduler_state_dict"])
    last_epoch = state_dict["epoch"]

    print(f"Model, optimizer and scheduler loaded from {path}.")
    return model, optimizer, scheduler, last_epoch
