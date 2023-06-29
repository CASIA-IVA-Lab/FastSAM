from ultralytics import YOLO
import gradio as gr
import torch
from utils.tools_gradio import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt
from PIL import ImageDraw
import numpy as np

# Load the pre-trained model
model = YOLO('./weights/FastSAM.pt')

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Description
title = "<center><strong><font size='8'>🏃 Fast Segment Anything 🤗</font></strong></center>"

news = """ # 📖 News
        🔥 2023/06/29: Support the text mode (Thanks for [gaoxinge](https://github.com/CASIA-IVA-Lab/FastSAM/pull/47)).

        🔥 2023/06/26: Support the points mode. (Better and faster interaction will come soon!)

        🔥 2023/06/24: Add the 'Advanced options" in Everything mode to get a more detailed adjustment.        
        """  

description_e = """This is a demo on Github project 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM). Welcome to give a star ⭐️ to it.
                
                🎯 Upload an Image, segment it with Fast Segment Anything (Everything mode). The other modes will come soon.
                
                ⌛️ It takes about 6~ seconds to generate segment results. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                
                🚀 To get faster results, you can use a smaller input size and leave high_visual_quality unchecked.
                
                📣 You can also obtain the segmentation results of any Image through this Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)
                
                😚 A huge thanks goes out to the @HuggingFace Team for supporting us with GPU grant.
                
                🏠 Check out our [Model Card 🏃](https://huggingface.co/An-619/FastSAM)
                
              """

description_p = """ # 🎯 Instructions for points mode
                This is a demo on Github project 🏃 [Fast Segment Anything Model](https://github.com/CASIA-IVA-Lab/FastSAM). Welcome to give a star ⭐️ to it.
                
                1. Upload an image or choose an example.
                
                2. Choose the point label ('Add mask' means a positive point. 'Remove' Area means a negative point that is not segmented).
                
                3. Add points one by one on the image.
                
                4. Click the 'Segment with points prompt' button to get the segmentation results.
                
                **5. If you get Error, click the 'Clear points' button and try again may help.**
                
              """

examples = [["examples/sa_8776.jpg"], ["examples/sa_414.jpg"], ["examples/sa_1309.jpg"], ["examples/sa_11025.jpg"],
            ["examples/sa_561.jpg"], ["examples/sa_192.jpg"], ["examples/sa_10039.jpg"], ["examples/sa_862.jpg"]]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_everything(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    text="",
    mask_random_color=True,
):
    input_size = int(input_size)  # 确保 imgsz 是整数
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)

    if len(text) > 0:
        results = format_results(results[0], 0)
        annotations, _ = text_prompt(results, text, input, device=device)
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data
    
    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)
    return fig


def segment_with_points(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    
    input_size = int(input_size)  # 确保 imgsz 是整数
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    
    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)

    global_points = []
    global_point_label = []
    return fig, None


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == 'Add Mask' else 0)
    
    print(x, y, label == 'Add Mask')
    
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image


cond_img_e = gr.Image(label="Input", value=default_example[0], type='pil')
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type='pil')
cond_img_t = gr.Image(label="Input with text", value="examples/dogs.jpg", type='pil')

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type='pil')
segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type='pil')
segm_img_t = gr.Image(label="Segmented Image with text", interactive=False, type='pil')

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(minimum=512,
                                         maximum=1024,
                                         value=1024,
                                         step=64,
                                         label='Input_size',
                                         info='Our model was trained on a size of 1024')

with gr.Blocks(css=css, title='Fast Segment Anything') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

        with gr.Column(scale=1):
            # News
            gr.Markdown(news)

    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()

            with gr.Column(scale=1):
                segm_img_e.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider.render()

                with gr.Row():
                    contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')

                    with gr.Column():
                        segment_btn_e = gr.Button("Segment Everything", variant='primary')
                        clear_btn_e = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=examples,
                            inputs=[cond_img_e],
                            outputs=segm_img_e,
                            fn=segment_everything,
                            cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    # text_box = gr.Textbox(label="text prompt")
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                        with gr.Column():
                            retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')

                # Description
                gr.Markdown(description_e)

    with gr.Tab("Points mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()
                
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(["Add Mask", "Remove Area"], value="Add Mask", label="Point_label (foreground/background)")

                    with gr.Column():
                        segment_btn_p = gr.Button("Segment with points prompt", variant='primary')
                        clear_btn_p = gr.Button("Clear points", variant='secondary')

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=examples,
                            inputs=[cond_img_p],
                            # outputs=segm_img_p,
                            # fn=segment_with_points,
                            # cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                # Description
                gr.Markdown(description_p)
                
    with gr.Tab("Text mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_t.render()

            with gr.Column(scale=1):
                segm_img_t.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider_t = gr.components.Slider(minimum=512,
                                         maximum=1024,
                                         value=1024,
                                         step=64,
                                         label='Input_size',
                                         info='Our model was trained on a size of 1024')
                with gr.Row():
                    with gr.Column():
                        contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')
                        text_box = gr.Textbox(label="text prompt", value="a black dog")

                    with gr.Column():
                        segment_btn_t = gr.Button("Segment with text", variant='primary')
                        clear_btn_t = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=["examples/dogs.jpg"],
                            inputs=[cond_img_e],
                            # outputs=segm_img_e,
                            # fn=segment_everything,
                            # cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                        with gr.Column():
                            retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')

                # Description
                gr.Markdown(description_e)
        
    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

    segment_btn_e.click(segment_everything,
                        inputs=[
                            cond_img_e,
                            input_size_slider,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                        ],
                        outputs=segm_img_e)

    segment_btn_p.click(segment_with_points,
                        inputs=[cond_img_p],
                        outputs=[segm_img_p, cond_img_p])
    
    segment_btn_t.click(segment_everything,
                        inputs=[
                            cond_img_t,
                            input_size_slider_t,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                            text_box,
                        ],
                        outputs=segm_img_t)

    def clear():
        return None, None
    
    def clear_text():
        return None, None, None
    
    clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_t.click(clear_text, outputs=[cond_img_p, segm_img_p, text_box])

demo.queue()
demo.launch()
