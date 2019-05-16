# YOLO3-TF
YOLO v3 re-implementation, and our code is based[stronger-yolo]{https://github.com/Stinky-Tofu/Stronger-yolo}

# Use backbone with VGG-16 and Mobilenet V2

## Usage
1. clone YOLO_v3 repository
    ``` bash
    git clone https://github.com/fanbinqi/YOLO3-TF.git
    ```
2. prepare data<br>
    (1) download datasets<br>
    Create a new folder named `data` in the directory where the `YOLO_V3` folder 
    is located, and then create a new folder named `VOC` in the `data/`.<br>
    Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and put datasets into `data/VOC`,
    name as `2012_trainval`、`2007_trainval`、`2007_test` separately. <br>
    The file structure is as follows:<br>
    |--YOLO_V3<br>
    |--data<br>
    |--|--VOC<br>
    |--|--|--2012_trainval<br>
    |--|--|--2007_trainval<br>
    |--|--|--2007_test<br>
    (2) convert data format<br>
    You should set `DATASET_PATH` in `config.py` to the path of the VOC dataset, for example:
    `DATASET_PATH = '/home/VOC'`,and then<br>
    ```bash
    python voc_annotation.py
    ```
3. prepare initial weights<br>
    Download [YOLOv3-608.weights](https://pjreddie.com/media/files/yolov3.weights) firstly, 
    put the yolov3.weights into `yolov3_to_tf/`, and then 
    ```bash
    cd yolov3_to_tf
    python3 convert_weights.py --weights_file=yolov3.weights --dara_format=NHWC --ckpt_file=./saved_model/yolov3_608_coco_pretrained.ckpt
    cd ..
    python rename.py
    ``` 
    if want to train model with MoBileNet V2, do not need to rename ckpt
    ```bash
    Download [mobilenet_v2_1.0_224.weights](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) firstly, 
    put the initial weight into `weights/`.
    python train.py
    ```

4. Train<br>
    ``` bash
    python train.py
    ```
5. Test<br>
    Download weight file [yolo_test.ckpt](https://drive.google.com/drive/folders/1We_P5L4nlLofR0IJJXzS7EEklZGUb9sz)<br>
    **If you want to get a higher mAP, you can set the score threshold to 0.01、use multi scale test、flip test.<br>
    If you want to use it in actual projects, or if you want speed, you can set the score threshold to 0.2.<br>**
    ``` bash
    python test.py --gpu=0 --map_calc=True --weights_file=model_path.ckpt
    cd mAP
    python main.py -na -np
    ```
