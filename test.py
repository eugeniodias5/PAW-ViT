import torch
from torch.cuda.amp import autocast

from data.dataloader import get_dataloader
from utils import (
    extract_feats_from_loader,
    evaluate_atrw,
    load_model,
    evaluate_CMC_map,
)
from models import make_model


def test(
    config,
    model=None,
    test_dl=None,
    query_dl=None,
    gallery_dl=None,
    device="cpu",
    logger=None,
):

    ds_name = config["DATASET"]["TEST"].lower()

    # build / load model
    if model is None:
        model = make_model(config, pretrained=True)
        if config["TRAIN"]["LOAD_MODEL"]:
            model = load_model(config["TRAIN"]["WEIGHTS_PATH"], model)
        else:
            msg = "Model not provided. Loading a pretrained model."
            (logger.info if logger else print)(msg)

    model.to(device)
    model.eval()

    if test_dl is None:
        if ds_name == "yak":
            if query_dl is None:
                query_dl = get_dataloader(config, mode="query")
            if gallery_dl is None:
                gallery_dl = get_dataloader(config, mode="gallery")
        elif test_dl is None:
            test_dl = get_dataloader(config, mode="test")

    fp16 = config["TRAIN"].get("FP16", False)
    # Scaler works only with CUDA devices
    use_amp = device != "cpu" and fp16

    with torch.no_grad():
        if ds_name == "yak":
            if use_amp:
                with autocast():
                    q_feats, q_ids = extract_feats_from_loader(
                        model=model,
                        dataloader=query_dl,
                        device=device,
                    )
                    g_feats, g_ids = extract_feats_from_loader(
                        model=model,
                        dataloader=gallery_dl,
                        device=device,
                    )
            else:
                q_feats, q_ids = extract_feats_from_loader(
                    model=model,
                    dataloader=query_dl,
                    device=device,
                )
                g_feats, g_ids = extract_feats_from_loader(
                    model=model,
                    dataloader=gallery_dl,
                    device=device
                )

            CMC, map = evaluate_CMC_map(
                query_features=q_feats,
                gallery_features=g_feats,
                query_labels=q_ids,
                gallery_labels=g_ids,
            )

            msg = (
                f"mAP: {map:.4f}, "
                f"Top-1: {CMC[0]:.4f}, Top-5: {CMC[4]:.4f}, Top-10: {CMC[9]:.4f}"
            )
            print(msg)
            if logger:
                logger.info(msg)
            return map

        elif ds_name == "atrw":
            if use_amp:
                with autocast():
                    feats, ids = extract_feats_from_loader(
                        model=model,
                        dataloader=test_dl,
                        device=device,
                        concat=True
                    )
            else:
                feats, ids = extract_feats_from_loader(
                    model=model,
                    dataloader=test_dl,
                    device=device,
                    concat=True
                )

            res = evaluate_atrw(
                feats,
                ids,
                config["DATASET"]["GT_LABELS_PATH"],
            )

            priv = res["result"][1]["private_split"]
            m1, m2 = priv["mAP(single_cam)"], priv["mAP(cross_cam)"]
            mmap = (m1 + m2) / 2
            msg = (
                f"mAP-single: {m1:.4f}, mAP-cross: {m2:.4f}, mMAP: {mmap:.4f}\n"
                f"Top-1 single: {priv['top-1(single_cam)']:.4f}, "
                f"Top-1 cross: {priv['top-1(cross_cam)']:.4f}\n"
                f"Top-5 single: {priv['top-5(single_cam)']:.4f}, "
                f"Top-5 cross: {priv['top-5(cross_cam)']:.4f}"
            )
            print(msg)
            if logger:
                logger.info(msg)
            return mmap

        else:
            raise ValueError("Invalid test dataset name.")
