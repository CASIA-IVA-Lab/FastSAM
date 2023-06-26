import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
# import clip


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    # transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):  # filte the overlap mask
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            # 将多个bbox合并成一个
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]

def fast_process(
    annotations,
    image,
    device,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    withContours=True,
    ):
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]

    original_h = image.height
    original_w = image.width
    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    if device == 'cpu':
        annotations = np.array(annotations)
        inner_mask = fast_show_mask(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            retinamask=use_retina,
            target_height=original_h,
            target_width=original_w,
        )
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
        inner_mask = fast_show_mask_gpu(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            retinamask=use_retina,
            target_height=original_h,
            target_width=original_w,
        )
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            if use_retina == False:
                annotation = cv2.resize(
                    annotation,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)

    image = image.convert('RGBA')
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), 'RGBA')
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if withContours:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), 'RGBA')
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image


#   CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    # 将annotation 按照面积 排序
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color == True:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))

    if retinamask == False:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return mask


def fast_show_mask_gpu(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    device = annotation.device
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # 找每个位置第一个非零值下标
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color == True:
        color = torch.rand((mask_sum, 1, 1, 3)).to(device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        ).to(device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    # 按index取数，index指每个位置选哪个batch的数，把mask_image转成一个batch的形式
    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # 使用向量化索引更新show的值
    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1
            )
        )
    if retinamask == False:
        mask_cpu = cv2.resize(
            mask_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    return mask_cpu


# # clip
# @torch.no_grad()
# def retriev(
#     model, preprocess, elements, search_text: str, device
# ) -> int:
#     preprocessed_images = [preprocess(image).to(device) for image in elements]
#     tokenized_text = clip.tokenize([search_text]).to(device)
#     stacked_images = torch.stack(preprocessed_images)
#     image_features = model.encode_image(stacked_images)
#     text_features = model.encode_text(tokenized_text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     probs = 100.0 * image_features @ text_features.T
#     return probs[:, 0].softmax(dim=0)


def crop_image(annotations, image_path):
    image = Image.open(image_path)
    ori_w, ori_h = image.size
    mask_h, mask_w = annotations[0]["segmentation"].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []
    # annotations, _ = filter_masks(annotations)
    # filter_id = list(_)
    for _, mask in enumerate(annotations):
        if np.sum(mask["segmentation"]) <= 100:
            filter_id.append(_)
            continue
        bbox = get_bbox_from_mask(mask["segmentation"])  # mask 的 bbox
        cropped_boxes.append(segment_image(image, bbox))  # 保存裁剪的图片
        # cropped_boxes.append(segment_image(image,mask["segmentation"]))
        cropped_images.append(bbox)  # 保存裁剪的图片的bbox

    return cropped_boxes, cropped_images, not_crop, filter_id, annotations


def box_prompt(masks, bbox, target_height, target_width):
    h = masks.shape[1]
    w = masks.shape[2]
    if h != target_height or w != target_width:
        bbox = [
            int(bbox[0] * w / target_width),
            int(bbox[1] * h / target_height),
            int(bbox[2] * w / target_width),
            int(bbox[3] * h / target_height),
        ]
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    # IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index


def point_prompt(masks, points, pointlabel, target_height, target_width):  # numpy 处理
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    if h != target_height or w != target_width:
        points = [
            [int(point[0] * w / target_width), int(point[1] * h / target_height)]
            for point in points
        ]
    onemask = np.zeros((h, w))
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation["segmentation"]
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                onemask += mask
            if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                onemask -= mask
    onemask = onemask >= 1
    return onemask, 0


# def text_prompt(annotations, args):
#     cropped_boxes, cropped_images, not_crop, filter_id, annotaions = crop_image(
#         annotations, args.img_path
#     )
#     clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
#     scores = retriev(
#         clip_model, preprocess, cropped_boxes, args.text_prompt, device=args.device
#     )
#     max_idx = scores.argsort()
#     max_idx = max_idx[-1]
#     max_idx += sum(np.array(filter_id) <= int(max_idx))
#     return annotaions[max_idx]["segmentation"], max_idx
