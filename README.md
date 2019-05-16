# YOLO3-TF
YOLO v3 re-implementation, and our code is based [stronger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo), a huge thank to him.

# Use backbone with VGG-16 and Mobilenet V2

## Usage
1. clone this repository
    ``` bash
    git clone https://github.com/fanbinqi/YOLO3-TF.git
    ```
2. prepare data<br>
    (1) download datasets<br>
    Create a new folder named `data` and then create a new folder named `VOC` in the `data/`.<br>
    Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar), and put datasets into `data/VOC`,
    name as `2012_trainval`、`2007_trainval`、`2007_test` separately. <br>
    The file structure is as follows:<br>
    |--YOLOV3-TF<br>
    |--data<br>
    |--|--VOC<br>
    |--|--|--2012_trainval<br>
    |--|--|--2007_trainval<br>
    |--|--|--2007_test<br>
    (2) convert data format<br>
    You should set `DATASET_PATH` in `config.py` to the path of the VOC dataset and then<br>
    ```bash
    python voc_annotation.py
    ```
3. prepare initial weights<br>
    3.1 yolov3
    Download [YOLOv3-608.weights](https://pjreddie.com/media/files/yolov3.weights) firstly, 
    put the yolov3.weights into `yolov3_to_tf/`, and then 
    ```bash
    cd yolov3_to_tf
    python3 convert_weights.py --weights_file=yolov3.weights --dara_format=NHWC -- ckpt_file=./saved_model/yolov3_608_coco_pretrained.ckpt
    cd ..
    python rename.py
    ``` 
    3.2 mobilenet v2
    if want to train model with MoBileNet V2
    ```bash
    Download [mobilenet_v2_1.0_224.weights](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) firstly, 
    put the initial weight into `weights/`.
    ```

4. Train<br>
    ``` bash
    python train.py
    ```
5. Test<br>
    ``` bash
    python test.py
    cd mAP
    python main.py
    ```

### If you are interested in this project, please QQ me (374873360)
