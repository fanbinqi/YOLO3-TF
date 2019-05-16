# coding:utf-8

import config as cfg
import numpy as np
import tensorflow as tf
from model.layer import *
from model.MobilenetV2 import MobilenetV2
from utils import utils


class YOLOV3(object):
    def __init__(self, training):
        self.__training = training
        self.__classes = cfg.CLASSES
        self.__num_classes = len(cfg.CLASSES)
        self.__strides = np.array(cfg.STRIDES)
        self.__gt_per_grid = cfg.ANCHOR_PER_SCALE
        self.__iou_loss_thresh = cfg.IOU_LOSS_THRESH
        self.__anchors = cfg.ANCHORS

    def build_nework(self, input_data, val_reuse=False):
        """
        :param input_data: shape为(batch_size, input_size, input_size, 3)
        :return: conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
        conv_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid * (5 + num_classes))
        conv_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid * (5 + num_classes))
        conv_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid * (5 + num_classes))
        conv_?是YOLO的原始卷积输出(raw_dx, raw_dy, raw_dw, raw_dh, raw_conf, raw_prob)
        pred_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid, 5 + num_classes)
        pred_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid, 5 + num_classes)
        pred_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid, 5 + num_classes)
        pred_?是YOLO预测bbox的信息(x, y, w, h, conf, prob)，(x, y, w, h)的大小是相对于input_size的
        """
        net_name = 'yolov3'
        with tf.variable_scope(net_name, reuse=val_reuse):
            feature_map_s, feature_map_m, feature_map_l = MobilenetV2(input_data, self.__training)

            conv = convolutional(name='conv0', input_data=feature_map_l, filters_shape=(1, 1, 1280, 512),
                                 training=self.__training)
            conv = separable_conv(name='conv1', input_data=conv, input_c=512, output_c=1024, training=self.__training)
            conv = convolutional(name='conv2', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training)
            conv = separable_conv(name='conv3', input_data=conv, input_c=512, output_c=1024, training=self.__training)
            conv = convolutional(name='conv4', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training)

            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lbbox = separable_conv(name='conv5', input_data=conv, input_c=512, output_c=1024,
                                        training=self.__training)
            conv_lbbox = convolutional(name='conv6', input_data=conv_lbbox,
                                       filters_shape=(1, 1, 1024, self.__gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox, anchors=self.__anchors[2],
                                num_classes=self.__num_classes, stride=self.__strides[2])
            # ----------**********---------- Detection branch of large object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)
            conv = upsample(name='upsample0', input_data=conv)
            conv = route(name='route0', previous_output=feature_map_m, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional('conv8', input_data=conv, filters_shape=(1, 1, 96 + 256, 256),
                                 training=self.__training)
            conv = separable_conv('conv9', input_data=conv, input_c=256, output_c=512, training=self.__training)
            conv = convolutional('conv10', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)
            conv = separable_conv('conv11', input_data=conv, input_c=256, output_c=512, training=self.__training)
            conv = convolutional('conv12', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)

            # ----------**********---------- Detection branch of middle object ----------**********----------
            conv_mbbox = separable_conv(name='conv13', input_data=conv, input_c=256, output_c=512,
                                        training=self.__training)
            conv_mbbox = convolutional(name='conv14', input_data=conv_mbbox,
                                       filters_shape=(1, 1, 512, self.__gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox, anchors=self.__anchors[1],
                                num_classes=self.__num_classes, stride=self.__strides[1])
            # ----------**********---------- Detection branch of middle object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv15', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)
            conv = upsample(name='upsample1', input_data=conv)
            conv = route(name='route1', previous_output=feature_map_s, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional(name='conv16', input_data=conv, filters_shape=(1, 1, 32 + 128, 128),
                                 training=self.__training)
            conv = separable_conv(name='conv17', input_data=conv, input_c=128, output_c=256, training=self.__training)
            conv = convolutional(name='conv18', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)
            conv = separable_conv(name='conv19', input_data=conv, input_c=128, output_c=256, training=self.__training)
            conv = convolutional(name='conv20', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)

            # ----------**********---------- Detection branch of small object ----------**********----------
            conv_sbbox = separable_conv(name='conv21', input_data=conv, input_c=128, output_c=256,
                                        training=self.__training)
            conv_sbbox = convolutional(name='conv22', input_data=conv_sbbox,
                                       filters_shape=(1, 1, 256, self.__gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_sbbox = decode(name='pred_sbbox', conv_output=conv_sbbox, anchors=self.__anchors[0],
                                num_classes=self.__num_classes, stride=self.__strides[0])
            # ----------**********---------- Detection branch of small object ----------**********----------
        for var in tf.global_variables(net_name):
            tf.add_to_collection(net_name, var)
        return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox

    def __focal(self, target, actual, alpha=1, gamma=2):
        focal = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal

    def __focal_loss(self, pre_conf, gamma=2, alph=0.25):
        focal = -alph * tf.pow((1 - pre_conf), gamma) * tf.log(pre_conf)
        return focal

    def __loss_per_scale(self, name, conv, pred, label, bboxes, anchors, stride):
        '''实际调用该函数的参数为：（'loss_sbbox', conv_sbbox, pred_sbbox, label_sbbox, sbboxes,self.__anchors[0], self.__strides[0]）
         :param name: loss的名字
         :param conv: conv是yolo卷积层的原始输出
         shape为(batch_size, output_size, output_size, anchor_per_scale * (5 + num_class))
         :param pred: conv是yolo输出的预测bbox的信息(x, y, w, h, conf, prob)，
         其中(x, y, w, h)的大小是相对于input_size的，如input_size=416，(x, y, w, h) = (120, 200, 50, 70)
         shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_class)
         :param label: shape为(batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes)
         只有best anchor对应位置的数据才为(x, y, w, h, 1, classes), (x, y, w, h)的大小是bbox纠正后的原始大小
         :param bboxes: shape为(batch_size, max_bbox_per_scale, 4)，
         存储的坐标为(x, y, w, h)，(x, y, w, h)的大小都是bbox纠正后的原始大小
         bboxes用于计算相应detector的预测框与该detector负责预测的所有bbox的IOU
         :param anchors: 相应detector的anchors
         :param stride: 相应detector的stride
         '''
        with tf.name_scope(name):
            conv_shape = tf.shape(conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]
            input_size = stride * output_size
            conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                     self.__gt_per_grid, 5 + self.__num_classes))
            conv_raw_dxdy = conv[:, :, :, :, 0:2]
            conv_raw_dwdh = conv[:, :, :, :, 2:4]
            conv_raw_conf = conv[:, :, :, :, 4:5]
            conv_raw_prob = conv[:, :, :, :, 5:]

            # (x, y, w, h, conf, prob)
            pred_xywh = pred[:, :, :, :, 0:4]
            pred_conf = pred[:, :, :, :, 4:5]

            # (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes)
            # (x, y, w, h, 1, classes)
            label_xy = label[:, :, :, :, 0:2]
            label_wh = label[:, :, :, :, 2:4]
            label_xywh = label[..., 0:4]
            respond_bbox = label[:, :, :, :, 4:5]
            label_prob = label[:, :, :, :, 5:]

            # (1)
            y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
            x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
            xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
            xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, self.__gt_per_grid, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)

            label_txty = 1.0 * label_xy / stride - xy_grid
            label_raw_twth = tf.log((1.0 * label_wh / stride) / anchors)
            label_raw_twth = tf.where(tf.is_inf(label_raw_twth), tf.zeros_like(label_raw_twth), label_raw_twth)

            input_size = tf.cast(input_size, tf.float32)

            bbox_loss_scale = 2.0 - 1.0 * label_wh[:, :, :, :, 0:1] * label_wh[:, :, :, :, 1:2] / (input_size ** 2)


            xy_loss = respond_bbox * bbox_loss_scale * \
                      tf.nn.sigmoid_cross_entropy_with_logits(labels=label_txty, logits=conv_raw_dxdy)
            wh_loss = 0.5 * respond_bbox * bbox_loss_scale * tf.square(
                label_raw_twth - conv_raw_dwdh)  #
            #
            # GIOU = utils.GIOU(pred_xywh, label_xywh)
            # GIOU = GIOU[..., np.newaxis]
            # input_size = tf.cast(input_size, tf.float32)
            # bbox_wh = label_xywh[..., 2:] - label_xywh[..., :2]
            # bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (input_size ** 2)
            # GIOU_loss = respond_bbox * bbox_loss_scale * (1.0 - GIOU)

            #
            iou = utils.iou_calc4(pred_xywh[:, :, :, :, np.newaxis, :],
                                  bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
            max_iou = tf.reduce_max(iou, axis=-1)
            max_iou = max_iou[:, :, :, :, np.newaxis]
            respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.__iou_loss_thresh, tf.float32)  # ¸ºÑù±¾

            conf_focal = self.__focal_loss(pred_conf, gamma=2, alph=0.25)

            conf_loss = conf_focal * (
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                    +
                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            )

            #
            prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
            loss = tf.concat([xy_loss, wh_loss, conf_loss, prob_loss], axis=-1)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))
            return loss

    def loss(self,conv_sbbox, conv_mbbox,
             conv_lbbox,pred_sbbox, pred_mbbox,
             pred_lbbox,label_sbbox, label_mbbox,
             label_lbbox,sbboxes, mbboxes, lbboxes):
        """
        :param conv_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale * (5 + num_classes))
        :param conv_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale * (5 + num_classes))
        :param conv_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale * (5 + num_classes))
        :param pred_sbbox: shape为(batch_size, image_size / 8, image_size / 8, anchors_per_scale, (5 + num_classes))
        :param pred_mbbox: shape为(batch_size, image_size / 16, image_size / 16, anchors_per_scale, (5 + num_classes))
        :param pred_lbbox: shape为(batch_size, image_size / 32, image_size / 32, anchors_per_scale, (5 + num_classes))
        :param label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, anchor_per_scale, 5 + num_classes)
        :param label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, anchor_per_scale, 5 + num_classes)
        :param label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, anchor_per_scale, 5 + num_classes)
        :param sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :param mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :param lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        :return:
        """
        # 形参为（name, conv, pred, label, bboxes, anchors, stride）
        loss_sbbox = self.__loss_per_scale('loss_sbbox', conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                           self.__anchors[0], self.__strides[0])
        loss_mbbox = self.__loss_per_scale('loss_mbbox', conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                           self.__anchors[1], self.__strides[1])
        loss_lbbox = self.__loss_per_scale('loss_lbbox', conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                           self.__anchors[2], self.__strides[2])
        with tf.name_scope('loss'):
            loss = loss_sbbox + loss_mbbox + loss_lbbox
        return loss