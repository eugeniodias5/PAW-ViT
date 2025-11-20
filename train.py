import os
import torch
import logging
import yaml

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from data.dataloader import get_dataloader
from loss.LossManager import LossManager
from optimizer.make_optimizer import make_optimizer
from models import make_model
from utils import save_model, load_model, compute_visibility
from test import test


def _setup_logger(results_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs or incorrect file paths
    if logger.handlers:
        logger.handlers.clear()

    # Define log file path dynamically based on results_path
    log_file = os.path.join(results_path, "log.txt")

    # Create file handler with formatter
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add the new handler to logger
    logger.addHandler(file_handler)

    return logger


def _finish_training(
    config,
    model,
    optimizer,
    scheduler,
    epoch,
    best_map,
    last_best_epoch,
    results_path,
    logger,
    trial=None,
):
    message = (
        "Training finished.\n"
        f"Best mAP: {best_map}\n"
        f"Best model saved at epoch {last_best_epoch}.\n"
        f"Last model saved at epoch {epoch}."
    )
    print(message)
    logger.info(message)

    last_model_path = os.path.join(results_path, f"last_model_epoch={epoch}.pth")
    if not trial:
        save_model(
            last_model_path,
            model,
            optimizer,
            scheduler,
            epoch,
        )
        print("Last model saved.")
        logger.info("Last model saved.")

    # Saving config file
    config_path = os.path.join(results_path, "config.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config, f)


def train(config, device="cpu", trial=None):
    
    if trial:
        # We set here the hyperparameters of the HP tuning
        hp_params = config["HP_PARAMETERS"]
        for param_category in hp_params.keys():
            for param_name in hp_params[param_category].keys():
                if "choices" in hp_params[param_category][param_name].keys():
                    config[param_category][param_name] = trial.suggest_categorical(
                        **hp_params[param_category][param_name]
                    )
                else:
                    config[param_category][param_name] = trial.suggest_float(
                        **hp_params[param_category][param_name]
                    )

        config["TRAIN"]["RES_PATH"] = f"trials/{trial.number}"

    ds_test_name = config["DATASET"]["TEST"].lower()

    # Get dataloaders
    train_dl = get_dataloader(config, mode="train")
    query_dl = None
    gallery_dl = None
    test_dl = None

    if ds_test_name == "yak":
        query_dl = get_dataloader(config, mode="query")
        gallery_dl = get_dataloader(config, mode="gallery")
    else:
        test_dl = get_dataloader(config, mode="test")

    num_classes = train_dl.dataset.get_num_classes()
    num_directions = train_dl.dataset.get_num_directions()
    num_directions = 1
    background_kpt = config["DATASET"].get("BACKGROUND_KPT", False)
    vis_threshold = config["DATASET"].get("VISIBILITY_THRESHOLD", None)

    # Create model
    local_train = False
    model_name = config["MODEL"]["BACKBONE"].lower()
    if "pawvit" in model_name and vis_threshold is not None:
        local_train = True
    else:
        # If the training is not local, we set the number of parts to 1
        config["DATASET"]["NUM_PARTS"] = 1

    model = make_model(
        config, num_classes=num_classes, num_directions=num_directions, pretrained=True
    )

    start_epoch = 0

    model.to(device)
    model.train()

    optimizer, scheduler = make_optimizer(config, model)

    if config["TRAIN"]["LOAD_MODEL"]:
        if config["TRAIN"]["RESUME"]:
            model, optimizer, scheduler, start_epoch = load_model(
                config["TRAIN"]["WEIGHTS_PATH"],
                model,
                optimizer,
                scheduler,
                is_train=True,
            )
        else:
            model = load_model(config["TRAIN"]["WEIGHTS_PATH"], model)

    id_devices = config["DEVICE_ID"]
    if len(id_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=id_devices)

    loss_fn = LossManager(config, local=local_train, device=device)

    # Prepare results path and logging
    results_path = config["TRAIN"]["RES_PATH"]
    os.makedirs(results_path, exist_ok=True)

    # Configure logging to file in results_path
    logger = _setup_logger(results_path)

    # If we are doing hyperparameter tuning, print the tuning parameters in the log
    if trial:
        msg = f"Trial {trial.number}"
        for param_category in hp_params.keys():
            for param_name in hp_params[param_category].keys():
                msg += f"\n{param_name}: {config[param_category][param_name]}"

        print(msg)
        logger.info(msg)

    best_map = 0.0
    last_best_epoch = 0
    eval_period = config["TRAIN"]["EVAL_PERIOD"]

    fp16 = config["TRAIN"].get("FP16", False)

    # Half precision
    use_amp = device != "cpu" and fp16
    # Scaler works only with CUDA devices
    scaler = GradScaler() if use_amp else None

    last_epoch = 0

    for epoch in range(start_epoch + 1, config["TRAIN"]["NUM_EPOCHS"] + 1):
        # set_seed(SEED + epoch)
        epoch_msg = f"Epoch {epoch}/{config['TRAIN']['NUM_EPOCHS']}"
        print(epoch_msg)
        logger.info(epoch_msg)

        model.train()
        scheduler.step(epoch=(epoch - 1))

        for batch in tqdm(
            train_dl, desc=f"Training lr = {optimizer.param_groups[0]['lr']:.2e}"
        ):
            optimizer.zero_grad()

            img, target, gt_masks, gt_directions = batch
            visibility = compute_visibility(
                gt_masks, threshold=vis_threshold, bckg_axis=background_kpt
            )

            img = img.to(device)
            output = model(img)


            if use_amp:
                with autocast():
                    if local_train and vis_threshold is not None:
                        feats, local_feats, logits, masks, local_directions = output

                        loss = loss_fn(
                            feats=feats,
                            local_feats=local_feats,
                            logits=logits,
                            target=target,
                            masks=masks,
                            gt_masks=gt_masks,
                            visibility=visibility,
                            local_directions=local_directions,
                            gt_directions=gt_directions,
                        )

                    else:
                        tensor, logits, directions = output
                        loss = loss_fn(
                            feats=tensor,
                            logits=logits,
                            target=target,
                            directions=directions,
                            gt_directions=gt_directions,
                        )

                # scale, backward, step, update
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                if local_train and vis_threshold is not None:
                    feats, logits, masks, local_directions = output

                    loss = loss_fn(
                        feats=feats,
                        local_feats=None,
                        logits=logits,
                        target=target,
                        masks=masks,
                        gt_masks=gt_masks,
                        visibility=visibility,
                        local_directions=local_directions,
                        gt_directions=gt_directions,
                    )

                else:
                    tensor, logits, directions = output
                    loss = loss_fn(
                        feats=tensor,
                        logits=logits,
                        target=target,
                        directions=directions,
                        gt_directions=gt_directions,
                    )
                loss.backward()
                optimizer.step()

        # Printing the loss
        avg_loss = loss_fn.get_epoch_loss()
        loss_msg = f"Epoch {epoch} Re-ID Loss: {avg_loss['reid_loss']} Seg Loss: {avg_loss['seg_loss']} Direction Loss: {avg_loss['direction_loss']} Total Loss: {avg_loss['total_loss']}"
        logger.info(loss_msg)
        print(loss_msg)

        if epoch % eval_period == 0:
            map_metric = test(
                config=config,
                model=model,
                device=device,
                logger=logger,
                query_dl=query_dl,
                gallery_dl=gallery_dl,
                test_dl=test_dl,
            )

            if map_metric > best_map:
                best_map = map_metric
                last_best_epoch = epoch

                # Erase models inside the results_path
                for f in os.listdir(results_path):
                    if "best_model" in f or "last_model" in f:
                        os.remove(os.path.join(results_path, f))

                model_save_path = os.path.join(
                    results_path, f"best_model_epoch={str(epoch)}.pth"
                )
                if not trial:
                    # If we are doing hyperparameter tuning, we don't save the model to save space
                    save_model(
                        model_save_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                    )
                best_msg = f"Best model saved at epoch {epoch} with mAP={map_metric}."
                print(best_msg)
                logger.info(best_msg)

        last_epoch = epoch

        early_stopping = int(config["TRAIN"]["EARLY_STOP"])
        if epoch - last_best_epoch >= early_stopping:
            stop_msg = f"No improvements after {early_stopping} epochs. Stopping training and saving last model."
            print(stop_msg)
            logger.info(stop_msg)
            break

    _finish_training(
        config,
        model,
        optimizer,
        scheduler,
        last_epoch,
        best_map,
        last_best_epoch,
        results_path,
        logger,
        trial,
    )
    return model, best_map
