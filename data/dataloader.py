from torch.utils.data import DataLoader
import torchvision.transforms as T

from data.dataset.ATRW import ATRW
from data.dataset.Yak import Yak

from data.sampler.triplet_sampler import RandomIdentitySampler
from data.transform.CustomTransform import *


SEED = 42


# Setting dataloader for reproducibility
def _makedataloader(dataset, batch_size, shuffle, num_workers, seed, sampler=None):
    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id):
        # Set the seed for each worker to ensure reproducibility
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        sampler=sampler,
    )


def _get_transform(
    crop_size, resize_size, mean, std, mode="train"
):
    if mode == "train":
        return CustomCompose(
            [
                CustomToTensor(),
                CustomResize(resize_size),
                CustomCrop(crop_size),
                CustomRandomRotation(degrees=15),
                CustomNormalize(mean=mean, std=std),
            ]
        )

    else:
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(resize_size),
                T.CenterCrop(crop_size),
                T.Normalize(mean=mean, std=std),
            ]
        )


def get_dataloader(config, mode="train"):
    train_ds_name = config["DATASET"]["TRAIN"].lower()
    test_ds_name = config["DATASET"]["TEST"].lower()

    if mode == "train":
        transform = _get_transform(
            crop_size=config["INPUT"]["CROP"],
            resize_size=config["INPUT"]["RESIZE"],
            mean=config["INPUT"]["MEAN"],
            std=config["INPUT"]["STD"],
            mode=mode,
        )

        # Define train dataset
        if train_ds_name == "atrw":
            train_dataset = ATRW(
                root=config["DATASET"]["ROOT"],
                mask_dir=config["DATASET"]["MASK_DIR"],
                transform=transform,
                mode=mode,
            )

        elif train_ds_name == "yak":
            train_dataset = Yak(
                root=config["DATASET"]["ROOT"],
                mask_dir=config["DATASET"]["MASK_DIR"],
                transform=transform,
                mode=mode,
            )

        else:
            raise ValueError("Invalid train dataset name.")

        sampler = RandomIdentitySampler(
            train_dataset,
            batch_size=config["DATASET"]["BATCH_SIZE"],
            num_instances=config["DATASET"]["NUM_INSTANCES"],
        )

        train_dl = _makedataloader(
            train_dataset,
            batch_size=config["DATASET"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=config["DATASET"]["NUM_WORKERS"],
            sampler=sampler,
            seed=SEED,
        )

        return train_dl

    transform = _get_transform(
        crop_size=config["INPUT"]["CROP"],
        resize_size=config["INPUT"]["RESIZE"],
        mean=config["INPUT"]["MEAN"],
        std=config["INPUT"]["STD"],
        mode=mode,
    )

    # Define test dataset
    if test_ds_name == "atrw":
        test_dataset = ATRW(
            root=config["DATASET"]["ROOT"],
            transform=transform,
            mode=mode,
        )

    elif test_ds_name == "yak":
        test_dataset = Yak(
            root=config["DATASET"]["ROOT"],
            transform=transform,
            mode=mode,
        )

    else:
        raise ValueError("Invalid test dataset name.")

    test_dl = _makedataloader(
        test_dataset,
        batch_size=config["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["DATASET"]["NUM_WORKERS"],
        seed=SEED + 777,
    )

    return test_dl
