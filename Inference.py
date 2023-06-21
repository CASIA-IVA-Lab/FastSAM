from ultralytics import YOLO
import sys
from utils.tools import *
import os
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../finally.pt', help='model')
    parser.add_argument('--img_path', type=str, default='./images/dogs.jpg',help='path to image file')
    parser.add_argument('--imgsz', type=int, default=1024,help='path to image file')
    parser.add_argument('--iou', type=float, default=0.9, help='iou threshold for filtering the annotations')
    parser.add_argument('--text_prompt', type=str, default=None,help='iou threshold for filtering the annotations')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output', type=str, default='./output/', help='image save path')
    parser.add_argument('--randomcolor', type=bool, default=True, help='mask random color')
    parser.add_argument('--point_prompt', type=str, default="[[0,0]]", help='[[x,y]]')
    parser.add_argument('--box_prompt', type=str, default="[0,0,0,0]", help='x,y,w,h')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)
    return parser.parse_args()



def main(args):
    # load model
    model = YOLO(args.model_path)
    args.point_prompt = eval(args.point_prompt)
    args.box_prompt = eval(args.box_prompt)
    results = model(args.img_path, imgsz=args.imgsz, device=args.device, retina_masks=True, iou=args.iou, conf=args.conf,
                         max_det=100)
    if args.box_prompt[2] != 0 and args.box_prompt[3] != 0:
        annotations = prompt(results, args,box=True)
        image_name = os.path.basename(args.img_path)
        post_process(annotations=annotations, image_path=args.img_path, mask_random_color=args.randomcolor,save_path=args.output, 
                     result_name=image_name,bbox=convert_box_xywh_to_xyxy(args.box_prompt))

    elif args.text_prompt!= None:
        results = format_results(results[0],0)
        annotations = prompt(results, args,text=True)
        image_name = os.path.basename(args.img_path)
        post_process(annotations=annotations, image_path=args.img_path, mask_random_color=args.randomcolor,
                     save_path=args.output, result_name=image_name)

    elif args.point_prompt[0] != [0,0]:
        results = format_results(results[0],0)
        annotations = prompt(results, args,point=True)
        image_name = os.path.basename(args.img_path)
        post_process(annotations=annotations, image_path=args.img_path, mask_random_color=args.randomcolor,save_path=args.output, 
                     result_name=image_name,points=args.point_prompt)

    else:
        results = format_results(results[0], 100)
        annotations = results
        #annotations, _ = filter_masks(results)
        image_name = os.path.basename(args.img_path)
        post_process(annotations=annotations, image_path=args.img_path, mask_random_color=args.randomcolor,save_path=args.output, result_name=image_name)
    
def prompt(results, args,box=None, point=None, text=None):
    annotations = []
    if box:
        mask, idx = box_prompt(results[0].masks.data, convert_box_xywh_to_xyxy(args.box_prompt))
    elif point:
        mask,idx = point_prompt(results, args.point_prompt)
    elif text:
        mask, idx = text_prompt(results, args)
    else:
        return annotations
    annotation = {}
    annotation['id'] = 0

    
    if box:
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = results[0].boxes.data[idx]
        annotation['score'] = results[0].boxes.conf[idx]
        annotation['area'] = annotation['segmentation'].sum()
    else:
        annotation['segmentation'] = mask
        annotation['bbox'] = results[idx]['bbox']
        annotation['score'] = results[idx]['score']
        annotation['area'] = annotation['segmentation'].sum()

    annotations.append(annotation)
    return annotations


if __name__ == '__main__':
    args = parse_args()
    main(args)