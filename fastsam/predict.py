# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
# Thanks for chenxwh.

import argparse
import cv2
import shutil
import ast
from cog import BasePredictor, Input, Path
from ultralytics import YOLO
from utils.tools import *


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {k: YOLO(f"{k}.pt") for k in ["FastSAM-s", "FastSAM-x"]}

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        model_name: str = Input(
            description="choose a model",
            choices=["FastSAM-x", "FastSAM-s"],
            default="FastSAM-x",
        ),
        iou: float = Input(
            description="iou threshold for filtering the annotations", default=0.7
        ),
        text_prompt: str = Input(
            description='use text prompt eg: "a black dog"', default=None
        ),
        conf: float = Input(description="object confidence threshold", default=0.25),
        retina: bool = Input(
            description="draw high-resolution segmentation masks", default=True
        ),
        box_prompt: str = Input(default="[0,0,0,0]", description="[x,y,w,h]"),
        point_prompt: str = Input(default="[[0,0]]", description="[[x1,y1],[x2,y2]]"),
        point_label: str = Input(default="[0]", description="[1,0] 0:background, 1:foreground"),
        withContours: bool = Input(
            description="draw the edges of the masks", default=False
        ),
        better_quality: bool = Input(
            description="better quality using morphologyEx", default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # default params

        out_path = "output"
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        args = argparse.Namespace(
            better_quality=better_quality,
            box_prompt=box_prompt,
            conf=conf,
            device=device,
            img_path=str(input_image),
            imgsz=1024,
            iou=iou,
            model_path="FastSAM-x.pt",
            output=out_path,
            point_label=point_label,
            point_prompt=point_prompt,
            randomcolor=True,
            retina=retina,
            text_prompt=text_prompt,
            withContours=withContours,
        )
        args.point_prompt = ast.literal_eval(args.point_prompt)
        args.box_prompt = ast.literal_eval(args.box_prompt)
        args.point_label = ast.literal_eval(args.point_label)

        model = self.models[model_name]

        results = model(
            str(input_image),
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=args.retina,
            iou=args.iou,
            conf=args.conf,
            max_det=100,
        )

        if args.box_prompt[2] != 0 and args.box_prompt[3] != 0:
            annotations = prompt(results, args, box=True)
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations,
                args=args,
                mask_random_color=args.randomcolor,
                bbox=convert_box_xywh_to_xyxy(args.box_prompt),
            )

        elif args.text_prompt != None:
            results = format_results(results[0], 0)
            annotations = prompt(results, args, text=True)
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations, args=args, mask_random_color=args.randomcolor
            )

        elif args.point_prompt[0] != [0, 0]:
            results = format_results(results[0], 0)
            annotations = prompt(results, args, point=True)
            # list to numpy
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations,
                args=args,
                mask_random_color=args.randomcolor,
                points=args.point_prompt,
            )

        else:
            fast_process(
                annotations=results[0].masks.data,
                args=args,
                mask_random_color=args.randomcolor,
            )

        out = "/tmp.out.png"
        shutil.copy(os.path.join(out_path, os.listdir(out_path)[0]), out)

        return Path(out)


def prompt(results, args, box=None, point=None, text=None):
    ori_img = cv2.imread(args.img_path)
    ori_h = ori_img.shape[0]
    ori_w = ori_img.shape[1]
    if box:
        mask, idx = box_prompt(
            results[0].masks.data,
            convert_box_xywh_to_xyxy(args.box_prompt),
            ori_h,
            ori_w,
        )
    elif point:
        mask, idx = point_prompt(
            results, args.point_prompt, args.point_label, ori_h, ori_w
        )
    elif text:
        mask, idx = text_prompt(results, args.text_prompt, args.img_path, args.device)
    else:
        return None
    return mask
