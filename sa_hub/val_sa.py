from ultralytics import YOLO
model = YOLO(model="FastSAM.pt", \
             )
model.val(data="sa.yaml", \
            epochs=100, \
            batch=1, \
            imgsz=1024, \
            device='0',\
            project='fastsam', \
            name='val', 
            val=False,
            save_json=True, \
            conf=0.001, \
            iou=0.9, \
            max_det=100, \
            )
