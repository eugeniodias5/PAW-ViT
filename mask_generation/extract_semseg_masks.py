# Script to extract pseudo-masks based on animal keypoints estimation
import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np

from ultralytics import YOLO


import numpy as np
from skimage.graph import route_through_array

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator

from torchvision import transforms
from transformers import AutoModelForImageSegmentation

import argparse
import  json, yaml
from tqdm import tqdm


MASK_SIZE = (64, 64)


transform_quadruped = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    """
    Loading image
    """
    def __init__(self, img_list, root_dir):
        self.img_list = img_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            img = Image.open(img_path).convert("RGB")
            return img_name, img, np.array(img) 
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            return None
    

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], [], []
    names, pils, nps = zip(*batch)
    return list(names), list(pils), list(nps)


def get_det_results(det_model, img, device='cuda'):
    results = det_model(img, verbose=False, strict=False)

    # Return the biggest bounding box
    biggest_bbox = None

    for detect in results:
        boxes = detect.boxes
        bbox = boxes.xyxy

        if len(bbox) == 0:
            continue

        bbox = np.array(bbox.cpu())

        # Consider only the biggest bounding box
        bboxes_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        biggest_bbox = bbox[np.argmax(bboxes_area)].reshape(1, 4)

    return biggest_bbox
    

def batch_segmentation(model, imgs, device, transform=None):
    """
    Batches segmentation by resizing all crops to input_size, 
    stacking them, and running one inference pass.
    """
    if not imgs:
        return []

    batch_tensors = []

    for img in imgs:
        batch_tensors.append(transform(img))

    batch_input = torch.stack(batch_tensors).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        preds = model(batch_input)[-1].sigmoid().cpu()

    # Resize masks back to original crop sizes
    final_masks = []
    for i, pred in enumerate(preds):
        w_orig, h_orig = imgs[i].size
        
        # Pred is (1, 1024, 1024), resize to (1, H_orig, W_orig)
        resize = transforms.Resize((h_orig, w_orig), interpolation=transforms.InterpolationMode.BILINEAR)
        mask_resized = resize(pred) 
        
        mask_numpy = mask_resized.squeeze(0).numpy()
        final_masks.append(mask_numpy)

    return final_masks


def get_segmentation_mask(image, bbox, model, transform, device='cuda'):

    seg_image = image.crop(bbox)
    
    if transform:
        # Passing bbox from numpy to a PIL image
        seg_image = transform(image)

    if len(seg_image.shape) == 3:
        seg_image = seg_image.unsqueeze(0)

    seg_image = seg_image.to(device)
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(seg_image)[-1].sigmoid().cpu()

    preds = preds[0]

    # Resize to the original image size
    resize = transforms.Resize((image.size[1], image.size[0]), interpolation=transforms.InterpolationMode.BILINEAR)
    mask = resize(preds)
    mask = mask.squeeze(0).numpy()
 
    return mask


def get_keypoints(crop_img, bbox, pose_model):
    # Convert image to numpy array
    crop_img = np.array(crop_img)
    
    pose_results = inference_topdown(
        pose_model,
        crop_img
    )

    keypoints = pose_results[0].pred_instances.keypoints
    conf = pose_results[0].pred_instances.keypoint_scores[0]

    # Limit keypoints to the image size
    keypoints = np.clip(keypoints, 0, [crop_img.shape[1] - 1, crop_img.shape[0] - 1])

    keypoints = np.squeeze(keypoints)
    conf = np.squeeze(conf)

    keypoints[:, 0] += bbox[0]
    keypoints[:, 1] += bbox[1]

    return keypoints, conf


def get_tail_labels(mask, start, end, num_points):
    # We find the shortest geodesic path between the two points and return labels between them
    cost_array = np.where(mask > 0.2, 1, np.inf)
    
    try:
        path, _ = route_through_array(cost_array, start, end, fully_connected=True, geometric=True)
    except ValueError:
        print("No minimal path found")
        return None

    path = np.array(path)

    # Find num_points between the two points
    if len(path) < num_points:
        return None
    
    indices = np.linspace(0, len(path) - 1, num_points + 2, dtype=int)
    # Ignoring the first and last points
    indices = indices[1:-1]

    path = path[indices]

    return path


def map_keypoints(ori_keypoints, ori_vis, tgt_keypoints, tgt_vis, mask, map_dict: dict, kpt_thresh=0.4):
    # This function should receive the original keypoints and, in the specified locations, map them to the target keypoints
    # It should also ignore some keypoints and create new connection keypoints
    
    # Mapping keypoints extracted using ap10k as reference to quadruped-80k
    kpts_map_dict = map_dict["kpts_map"]
    for idx in kpts_map_dict.keys():
        ori_keypoints[int(idx)] = tgt_keypoints[int(kpts_map_dict[idx])]
        ori_vis[int(idx)] = tgt_vis[int(kpts_map_dict[idx])]

    # Ignoring keypoints
    ignore_dict = map_dict["kpts_ignore"]
    for idx in ignore_dict:
        # Setting to zero the visibility of the ignored keypoints
        ori_vis[int(idx)] = 0

    kpts_labels = []

    # Extracting the keypoints classification labels
    kpts_indices = map_dict["kpts_description"].keys()

    for idx in sorted(kpts_indices, key=lambda x: int(x)):
        kpts_labels.append(map_dict["kpts_description"][idx]["type"])

    # Converting the kpts_indices from string to int
    mapped_kpts_indices = dict(zip(set(kpts_labels), range(len(kpts_labels))))
    kpts_indices = map(lambda x: mapped_kpts_indices[x], kpts_labels)
    kpts_indices = list(kpts_indices)

    # Creating new keypoints in the middle of some specified loations
    connections = map_dict["connections"]

    for connect in connections:
        c_link = connect["link"]
        c_class = connect["class"]

        kpt_1, kpt_2 = c_link

        # Checking visibility
        if ori_vis[int(kpt_1)] > kpt_thresh and ori_vis[int(kpt_2)] > kpt_thresh:
            # Getting the coordinates of the two keypoints
            x1, y1 = ori_keypoints[int(kpt_1)]
            x2, y2 = ori_keypoints[int(kpt_2)]

            # Getting the coordinates of the new keypoint
            x_new = (x1 + x2) / 2
            y_new = (y1 + y2) / 2

            # Adding the new keypoint to the list
            ori_keypoints = np.vstack((ori_keypoints, [x_new, y_new]))
            ori_vis = np.hstack((ori_vis, [1]))

            kpts_labels.append(c_class)
    

    # Handling tail keypoints if they exist
    if ori_vis[22] > kpt_thresh and ori_vis[23] > kpt_thresh:
        start_tail = (int(ori_keypoints[23][0]), int(ori_keypoints[23][1]))       
        end_tail = (int(ori_keypoints[22][0]), int(ori_keypoints[22][1]))

        # Check if end_tail is inside the mask, if it is not replace by the closest point inside the mask
        if mask[end_tail[1], end_tail[0]] < 0.2:
            # Find the closest point inside the mask
            mask_indices = np.where(mask > 0.2)
            distances = np.sqrt((mask_indices[0] - end_tail[1]) ** 2 + (mask_indices[1] - end_tail[0]) ** 2)
            closest_index = np.argmin(distances)
            end_tail = (mask_indices[1][closest_index], mask_indices[0][closest_index])

        if mask[start_tail[1], start_tail[0]] < 0.2:
            # Find the closest point inside the mask
            mask_indices = np.where(mask > 0.2)
            distances = np.sqrt((mask_indices[0] - start_tail[1]) ** 2 + (mask_indices[1] - start_tail[0]) ** 2)
            closest_index = np.argmin(distances)
            start_tail = (mask_indices[1][closest_index], mask_indices[0][closest_index])

        inv_mask = np.transpose(mask, (1, 0))
        
        tail_labels = get_tail_labels(inv_mask, start_tail, end_tail, 2)

        if tail_labels is not None:
            for tail_label in tail_labels:
                x, y = tail_label
                ori_keypoints = np.vstack((ori_keypoints, [x, y]))
                ori_vis = np.hstack((ori_vis, [1]))

                kpts_labels.append(kpts_labels[23])
            
    # Converting the kpts_labels to a numpy array
    kpts_labels = np.array(kpts_labels)
    
    return ori_keypoints, ori_vis, kpts_labels


def vis_keypoints(img, keypoints, conf, dataset_info=None, kpt_thresh=0.4, save_kpt=None, tgt_size=(448, 448)):
    if dataset_info:
        dataset_info = dataset_info["keypoint_info"]

    # Convert the image to a numpy array
    img = np.array(img)
    
    # Draw a circle for each keypoint
    for i in range(conf.shape[0]):
        if conf[i] < kpt_thresh:
            continue

        x, y = keypoints[i]
        cv.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)

        if dataset_info:
            cv.putText(img, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # BGR to RGB conversion
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize to target size
    img = cv.resize(img, (tgt_size[0], tgt_size[1]), interpolation=cv.INTER_LINEAR)
    
    if save_kpt:
        cv.imwrite(save_kpt, img)

    cv.imshow("Keypoints", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def vis_masks(img, masks, save_mask=None):
    """
    Overlays each body part's mask on the image using a distinct color for each.
    Parameters:
      img: The input image.
      masks: A numpy array of shape (K, 64, 64) containing the masks for K body parts.
    """
    img = np.array(img)
    masks = np.array(masks)
    
    # Get image size
    img_height, img_width = img.shape[:2]
    
    colors = [
        (255, 0, 0),    # Blue-ish
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 255)   # Purple
    ]
 
    # Create an empty overlay image (float for accumulation)
    overlay = np.zeros_like(img, dtype=np.float32)
    
    # Process each body part's mask separately.
    for i in range(masks.shape[0]):
        mask = masks[i]
        # Resize the mask to the original image size.
        mask_resized = cv.resize(mask, (img_width, img_height), interpolation=cv.INTER_LINEAR)

        # We put to 0 negative values and normalize the mask to range [0, 1]
        mask_resized[mask_resized < 0] = 0
        mask_resized[mask_resized > 1] = 1
        mask_norm = cv.normalize(mask_resized, None, 0, 1.0, cv.NORM_MINMAX)
        
        # Get the color for this part (BGR)
        color = colors[i % len(colors)]
        
        # Create a colored version of the mask.
        colored_mask = np.zeros_like(img, dtype=np.float32)
        colored_mask[..., 0] = mask_norm * color[0]
        colored_mask[..., 1] = mask_norm * color[1]
        colored_mask[..., 2] = mask_norm * color[2]
        
        # Accumulate the colored mask in the overlay.
        overlay += colored_mask
    
    # Clip the overlay to valid pixel range and convert to uint8.
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Blend the original image with the overlay.
    overlayed_img = cv.addWeighted(img, 0.5, overlay, 0.5, 0)

    if save_mask:
        cv.imwrite(save_mask, overlayed_img)
    
    cv.imshow("Overlayed Image", overlayed_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_masks(masks, save_path):
    # Check if the directory exists, if not create it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the masks as a numpy file
    np.save(save_path, masks)


def compute_masks(keypoints, vis, labels, mask, kpt_thresh=0.4, mask_size=(64, 64), background=False):
    
    H, W = mask_size

    # We reshape the keypoints coordinates to the mask size
    norm_keypoints = keypoints.copy()
    norm_keypoints[:, 0] = (norm_keypoints[:, 0] / mask.shape[1]) * W
    norm_keypoints[:, 1] = (norm_keypoints[:, 1] / mask.shape[0]) * H
    

    # We now reshape the mask to the desired mask size
    mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
  
    # First, for each pixel where the mask is not zero, we calculate the minimal distance to each body part
    unique_labels = np.unique(labels)
    dist_map = np.zeros((mask.shape[0], mask.shape[1], len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        # Get the coordinates of the keypoints with the specified label
        kpt_coords = norm_keypoints[labels == label]
        kpt_vis = vis[labels == label]

        # Calculate the distance from each pixel to the keypoints
        temp_dist = np.zeros((mask.shape[0], mask.shape[1], len(kpt_coords)))
        for j, (kpt, kpt_v) in enumerate(zip(kpt_coords, kpt_vis)):
            if kpt_v < kpt_thresh:
                temp_dist[:, :, j] = np.inf
                continue

            x, y = kpt
            
            temp_dist[:, :, j] = np.sqrt((np.arange(mask.shape[0])[:, None] - y) ** 2 + (np.arange(mask.shape[1])[None, :] - x) ** 2)

        # Get the minimal distance
        dist_map[:, :, i] = temp_dist.min(axis=2)
        
    # Now we convert the distance map to a probability map
    kernels = np.zeros((mask.shape[0], mask.shape[1], len(unique_labels)))

    # We find the minimum distance and set it to 1, and the other distances to 0
    # Find the minimum in axis 2
    min_dist = dist_map.min(axis=2)
    
    # Set the minimum distance to 1 and the others to 0
    for i in range(kernels.shape[2]):
        kernels[:, :, i] = np.where(dist_map[:, :, i] == min_dist, 1, 0)

    # Normalize the kernels to sum to 1
    sum_pixels = kernels.sum(axis=-1, keepdims=True)
    masks = kernels / (sum_pixels + 1e-8)
    
    # Convert to 0 the pixels where the mask is zero
    masks[mask < 0.2] = 0

    if background:
        # We create the background axis where the mask probability is less than 0
        background_mask = np.zeros_like(masks[:, :, -1])
        background_mask[mask < 0.2] = 1
        background_mask[mask >= 0.2] = 0

        background_mask = np.expand_dims(background_mask, axis=-1)
        # We concatenate the background mask to the kernels
        masks = np.concatenate((masks, background_mask), axis=-1)

    # Transpose the kernels
    masks = np.transpose(masks, (2, 0, 1))
     
    return masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for YOLO and Seg.')
    return parser.parse_args()


if __name__ == "__main__":
    # Reading configuration file from path in --config
    args = parse_args()

    # Loading configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Defining the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    
    # Loading detection model
    det_model = YOLO(config["WEIGHTS"]["DET"]).to(device)
    det_model.eval()

    seg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    seg_model.to(device)
    seg_model.eval()

    # Load pose estimator
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    
    pose_estimator_quadruped = init_pose_estimator(
        config["CONFIG"]["POSE_QUADRUPED"],
        config["WEIGHTS"]["POSE_QUADRUPED"],
        device=device,
        cfg_options=cfg_options
    )
    pose_estimator_quadruped.eval()
    
    pose_estimator_ap10k = init_pose_estimator(
        config["CONFIG"]["POSE_AP10K"],
        config["WEIGHTS"]["POSE_AP10K"],
        device=device,
        cfg_options=cfg_options
    )
    pose_estimator_ap10k.eval()

    imgs_list = []

    # Reading annotation txt file
    with open(config["DATASET"]["ANN"], 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:

            img_name = line.split(" ")[0]
            imgs_list.append(img_name)

    # Defining dataloader
    dataset = ImageDataset(imgs_list, config["DATASET"]["PATH"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    with open(config["DATASET"]["MAPPING"], 'r') as f:
        mapping = json.load(f)
    
    save_masks_path = config["DATASET"]["OUTPUT_MASKS"]
    os.makedirs(save_masks_path, exist_ok=True)

    num_empty = 0

    for batch_names, batch_imgs_pil, batch_imgs_np in tqdm(loader, desc="Processing"):
        det_results = det_model(batch_imgs_pil, verbose=False, stream=False)

        imgs = batch_imgs_pil
        bboxes = []
        orig_sizes = [(img.size[0], img.size[1]) for img in imgs]

        for i, detect in enumerate(det_results):
            boxes = detect.boxes.xyxy.cpu().numpy()

            if len(boxes) == 0:
                w, h = imgs[i].size
                bbox = [[0, 0, w, h]]
            else:
                # Take the biggest bounding box
                bbox_array = np.array(boxes)
                bboxes_area = (bbox_array[:, 2] - bbox_array[:, 0]) * (bbox_array[:, 3] - bbox_array[:, 1])
                bbox = bbox_array[np.argmax(bboxes_area)].reshape(1, 4)
            
            bboxes.append(bbox[0])
        
        # Batch segmentation
        masks = batch_segmentation(seg_model, imgs, device=device, transform=transform_quadruped)

        # We process image by image since inference_topdown does not support batch processing
        for k, img_name in enumerate(batch_names):
            bbox = bboxes[k]
            bbox = [int(b) for b in bbox]

            crop = imgs[k].crop(bbox)

            mask = masks[k]

            keypoints_quadruped, conf_quadruped = get_keypoints(crop, bbox, pose_estimator_quadruped)
            keypoints_ap10k, conf_ap10k = get_keypoints(crop, bbox, pose_estimator_ap10k)

            keypoints, vis, labels = map_keypoints(
                keypoints_quadruped, conf_quadruped, keypoints_ap10k, conf_ap10k, mask=mask, map_dict=mapping, kpt_thresh=config["THRESH"]["QUADRUPED"]
            )

            masks_final = compute_masks(keypoints, vis, labels, mask, kpt_thresh=config["THRESH"]["QUADRUPED"], mask_size=MASK_SIZE, background=config["BACKGROUND_MASK"])          
            
            if not (masks > 0.4).any():
                print(f"Empty mask for image {img_name}.")
                num_empty += 1

            # Save the masks
            save_masks(masks_final, os.path.join(save_masks_path, img_name.split(".")[0] + ".npy"))

    print(f"Number of empty maks: {num_empty} out of {len(imgs_list)}")
    