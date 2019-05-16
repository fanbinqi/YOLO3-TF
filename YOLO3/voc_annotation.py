#coding: utf-8

import os
import xml.etree.ElementTree as ET
import config as cfg

def convert_voc_annotation(data_path, data_type, annotation_path, use_difficult_bbox=True):
    '''
    :param data_path:
    :param data_type:  训练或是测试
    :param annotation_path:
    :param use_difficult_bbox:
    :return:
    '''
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with file(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with file(annotation_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            annotation += '\n'
            print(annotation)
            f.write(annotation)
    return len(image_inds)


if __name__ == '__main__':
    train_annotation_path = os.path.join(cfg.ANNOTATION_PATH, 'train_annotation.txt')
    test_annotation_path = os.path.join(cfg.ANNOTATION_PATH, 'test_annotation.txt')
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)
    num1 = convert_voc_annotation(os.path.join(cfg.DATASET_PATH, '2012_trainval'), 'trainval', train_annotation_path,
                                  False)
    num2 = convert_voc_annotation(os.path.join(cfg.DATASET_PATH, '2007_trainval'), 'trainval', train_annotation_path,
                                  False)
    num3 = convert_voc_annotation(os.path.join(cfg.DATASET_PATH, '2007_test'), 'test', test_annotation_path, False)
    print('The number of image for train is:'.ljust(50), num1 + num2)
    print('The number of image for test is:'.ljust(50), num3)