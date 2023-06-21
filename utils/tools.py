
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import torch
import os
import clip

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
    segmented_image_array[y1:y2,x1:x2] = image_array[y1:y2,x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    #transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros((image_array.shape[0],image_array.shape[1]), dtype=np.uint8)
    transparency_mask[y1:y2,x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def format_results(result,filter = 0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

    
        if torch.sum(mask) < filter:
            continue
        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def filter_masks(annotations):       # filte the overlap mask
    annotations.sort(key=lambda x: x['area'], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b['area'] < a['area']:
                    if (a['segmentation'] & b['segmentation']).sum() / b['segmentation'].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove],to_remove

def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def show_mask(annotation, ax,random_color=False,bbox=None,points=None):
    if random_color :    # random mask color
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    if type(annotation) == dict:
        annotation = annotation['segmentation']
    mask = annotation
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # draw box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
    # draw point
    if points is not None:
        ax.scatter([point[0] for point in points], [point[1] for point in points], s=10, c='g')
    ax.imshow(mask_image)
    return mask_image

def post_process(annotations,image_path,mask_random_color,save_path,result_name,bbox=None,points=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(annotations):
        show_mask(mask, plt.gca(),random_color=mask_random_color,bbox=bbox,points=points)
    plt.axis('off')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, result_name), bbox_inches='tight', pad_inches=0.0)


# clip
@torch.no_grad()
def retriev(model,preprocess,elements: [Image.Image], search_text: str,device) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def crop_image(annotations,image_path):
    image = Image.open(image_path)
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []
    #annotations, _ = filter_masks(annotations)
    #filter_id = list(_)
    for _, mask in enumerate(annotations):
        if np.sum(mask["segmentation"]) <= 100:
            filter_id.append(_)
            continue
        bbox = get_bbox_from_mask(mask["segmentation"])     # mask 的 bbox
        cropped_boxes.append(segment_image(image, bbox))              # 保存裁剪的图片
        #cropped_boxes.append(segment_image(image,mask["segmentation"]))
        cropped_images.append(bbox)                         # 保存裁剪的图片的bbox

    return cropped_boxes, cropped_images, not_crop, filter_id, annotations

def box_prompt(masks, bbox):
    h = masks.shape[1]
    w = masks.shape[2]
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    #IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = ((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))

    masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index], max_iou_index

def point_prompt(masks, points): # numpy 处理
    onemask = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation
        for point in points:
            if mask[point[1],point[0]]==1:
                onemask += mask
            
    onemask = onemask.clip(0, 1)
    return onemask,0


def text_prompt(annotations, args):
    cropped_boxes, cropped_images, not_crop, filter_id, annotaions = crop_image(annotations, args.img_path)
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    scores = retriev(clip_model, preprocess, cropped_boxes, args.text_prompt, device=args.device)
    max_idx = scores.argsort()
    max_idx = max_idx[-1]
    max_idx += sum(np.array(filter_id) <= int(max_idx))
    return annotaions[max_idx]["segmentation"], max_idx
