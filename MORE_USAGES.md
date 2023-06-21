# MORE_USAGES



### Everything mode
```angular2html
python Inference.py --model_path ./weights/FastSAM.pt \
                    --img_path ./images/dogs.jpg \
                    --imgsz 720 \
```
![everything mode](assets/more_usages/everything_mode.png)



### use more points
The mask of the foreground point, labeled as 0 and represented by the yellow color, will be displayed, while the mask of the background point, labeled as 1 and indicated by the purple color, will be suppressed.
```angular2html
python Inference.py --model_path ./weights/FastSAM.pt \
                    --img_path ./images/dogs.jpg  \
                    --point_prompt "[[520,360],[620,300],[520,300],[620,360]]" \
                    --point_label "[1,0,1,0]"
```
![points prompt](assets/more_usages/more_points.png)
### draw mask edge
use `--withContours True` to draw the edge of the mask,
when `--better_quality True` is set, the edge will be more smooth
```angular2html
python Inference.py --model_path ./weights/FastSAM.pt \
                    --img_path ./images/dogs.jpg \  
                    --point_prompt "[[620,360]]" \
                    --point_label "[1]" \
                    --withContours True \
                    --better_quality True
```
![Draw Edge](assets/more_usages/draw_edge.png)
### use box prompt
use `--box_prompt [x,y,w,h]` to specify the bounding box of the foreground object
```angular2html
python Inference.py --model_path ./weights/FastSAM.pt \
                    --img_path ./images/dogs.jpg \
                    --box_prompt [570,200,230,400] \
```
![box prompt](assets/more_usages/box_prompt.png)

### use text prompt
use `--text_prompt "text"` to specify the text prompt
```angular2html
python Inference.py --model_path ./weights/FastSAM.pt \
                    --img_path ./images/cat.jpg \
                    --text_prompt "cat" \
                    --better_quality True \
                    --withContours True 
```
![text prompt](assets/more_usages/text_prompt_cat.png)