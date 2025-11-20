from models.seresnet import SeResNet50
from models.Vit import (
    vit_base_patch16_224,
    vit_base_patch16_384,
    vit_large_patch16_224,
    vit_small_patch16_224,
    vit_base_seresnet50_patch16_224,
)
from models.Pawvit import (
    pawvit_small_patch16_224,
    pawvit_base_patch16_224,
    pawvit_base_patch16_384,
    pawvit_base_patch32_384,
    pawvit_base_seresnet50_patch16_224,
    pawvit_large_patch16_224,
)


def make_model(config, num_classes=1000, num_directions=1, pretrained=None):
    model_name = config["MODEL"]["BACKBONE"].lower()
    use_direction = True if config["LOSS"]["DIRECTION"] else False
    heatmap_size = config["MODEL"]["HEATMAP_SIZE"]
    patch_size = config["MODEL"]["PATCH_SIZE"]

    if pretrained:
        pretrained = config["MODEL"]["PRETRAINED"]

    img_size = (config["INPUT"]["CROP"][0], config["INPUT"]["CROP"][1])

    if model_name in ["se_resnet50", "seresnet50"]:
        model = SeResNet50(
            num_parts=config["DATASET"]["NUM_PARTS"],
            pretrained=pretrained,
            num_classes=num_classes,
            use_direction=use_direction,
            num_directions=num_directions,
            use_bnneck=config["MODEL"]["BNNECK"],
        )

    elif "vit" in model_name:
        if "vit_small" in model_name:
            if "paw" in model_name:
                vit_func = pawvit_small_patch16_224
            else:
                vit_func = vit_small_patch16_224

        if "vit_base" in model_name:
            if "seresnet50" in model_name:
                if "paw" in model_name:
                    vit_func = pawvit_base_seresnet50_patch16_224
                else:
                    vit_func = vit_base_seresnet50_patch16_224

            elif "384" in model_name:
                if "paw" in model_name:
                    if patch_size == 32:
                        vit_func = pawvit_base_patch32_384
                    else:
                        vit_func = pawvit_base_patch16_384
                else:
                    vit_func = vit_base_patch16_384

            else:
                if "paw" in model_name:
                    vit_func = pawvit_base_patch16_224
                else:
                    vit_func = vit_base_patch16_224

        elif "vit_large" in model_name:
            if "paw" in model_name:
                vit_func = pawvit_large_patch16_224
            else:
                vit_func = vit_large_patch16_224

        else:
            raise NotImplementedError(f"Model {model_name} is not implemented.")

        model = vit_func(
            num_parts=config["DATASET"]["NUM_PARTS"],
            patch_size=config["MODEL"]["PATCH_SIZE"],
            stride_size=config["MODEL"]["STRIDE_SIZE"],
            pretrained=pretrained,
            num_classes=num_classes,
            num_directions=num_directions,
            img_size=img_size,
            use_bnneck=config["MODEL"]["BNNECK"],
            background_kpt=config["DATASET"]["BACKGROUND_KPT"],
            heatmap_size=heatmap_size,
        )

    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")

    return model
