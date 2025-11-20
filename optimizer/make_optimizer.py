import torch

from optimizer.cosine_scheduler import CosineLRScheduler


def _make_lr_scheduler(config, optimizer):
    scheduler_name = config["SCHEDULER"]["NAME"]

    if scheduler_name.lower() == "cosineannealingwarmrestarts":
        t_0 = config["TRAIN"]["NUM_EPOCHS"]
        lr_min = 0.0
        if config["SCHEDULER"]["LR_MIN_RATIO"]:
            lr_min = config["OPTIMIZER"]["LR"] * config["SCHEDULER"]["LR_MIN_RATIO"]
        else:
            lr_min = config["SCHEDULER"]["LR_MIN"]

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=config["TRAIN"]["NUM_EPOCHS"], eta_min=lr_min
        )

    elif scheduler_name.lower() == "cosinelrscheduler":
        # Defining the minimum learning rate
        lr_min = 0.0
        if config["SCHEDULER"]["LR_MIN_RATIO"]:
            lr_min = config["OPTIMIZER"]["LR"] * config["SCHEDULER"]["LR_MIN_RATIO"]

        warmup_lr_init = 0.0
        if config["SCHEDULER"]["WARMUP_LR_INIT_RATIO"]:
            warmup_lr_init = (
                config["OPTIMIZER"]["LR"] * config["SCHEDULER"]["WARMUP_LR_INIT_RATIO"]
            )

        warmup_epochs = 0
        if config["SCHEDULER"]["WARMUP_EPOCHS_RATIO"]:
            warmup_epochs = (
                config["TRAIN"]["NUM_EPOCHS"]
                * config["SCHEDULER"]["WARMUP_EPOCHS_RATIO"]
            )

        num_cycles = 1
        if config["SCHEDULER"]["NUM_CYCLES"]:
            num_cycles = config["SCHEDULER"]["NUM_CYCLES"]

        decay_rate = 0
        if config["SCHEDULER"]["DECAY_RATE"]:
            decay_rate = config["SCHEDULER"]["DECAY_RATE"]

        num_epochs = config["TRAIN"]["NUM_EPOCHS"]
        t_initial = int(num_epochs - warmup_epochs) / num_cycles

        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=t_initial,
            t_mul=1,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_epochs,
            warmup_prefix=True,
            cycle_limit=num_cycles,
            decay_rate=decay_rate,
            t_in_epochs=True,
        )

    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} is not implemented")

    return scheduler


def make_optimizer(config, model):
    optimizer_name = config["OPTIMIZER"]["NAME"]
    base_lr = config["OPTIMIZER"]["LR"]
    weight_decay = config["OPTIMIZER"]["WEIGHT_DECAY"]

    if "LR_CLASSIFIER_RATIO" in config["OPTIMIZER"]:
        classifier_lr = base_lr * config["OPTIMIZER"]["LR_CLASSIFIER_RATIO"]
        decoder_lr = base_lr * config["OPTIMIZER"]["LR_DECODER_RATIO"]
        agg_attn_lr = base_lr * config["OPTIMIZER"]["LR_AGG_ATTN_RATIO"]

        # Separate classifier parameters from base parameters
        classifier_params = []
        decoder_params = []
        agg_attn_params = []
        base_params = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if "fc" in n and ("weight" in n or "bias" in n):
                classifier_params.append(p)

            elif "pose_decoder" in n or "pose_decoders" in n:
                decoder_params.append(p)

            elif ("agg_attn" in n) or ("agg_block" in n) or ("agg_token" in n):
                agg_attn_params.append(p)

            else:
                base_params.append(p)

        param_groups = [
            {"params": base_params, "lr": base_lr},
            (
                {"params": classifier_params, "lr": classifier_lr}
                if classifier_params
                else None
            ),
            {"params": decoder_params, "lr": decoder_lr} if decoder_params else None,
            {"params": agg_attn_params, "lr": agg_attn_lr} if agg_attn_params else None,
        ]
        param_groups = [g for g in param_groups if g is not None]
    else:
        param_groups = model.parameters()

    if optimizer_name.lower() == "sgd":
        optim = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=config["OPTIMIZER"]["MOMENTUM"],
            weight_decay=weight_decay,
        )

    elif optimizer_name.lower() == "adam":
        optim = torch.optim.Adam(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name.lower() == "adamw":
        optim = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )

    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented")

    if config["SCHEDULER"]:
        scheduler = _make_lr_scheduler(config, optim)
        return optim, scheduler

    else:
        raise ValueError(
            "Scheduler is not defined. Please define a scheduler in the configuration file"
        )
