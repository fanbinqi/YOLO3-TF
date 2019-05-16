# coding: utf-8

from model.yolov3 import YOLOV3
import config as cfg
from data import Data
import tensorflow as tf
import numpy as np
import os
import logging
import time


class YoloTrain(object):
    def __init__(self):
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__learn_rate_init = cfg.LEARN_RATE_INIT
        self.__learn_rate_end = cfg.LEARN_RATE_END
        self.__max_periods = cfg.MAX_PERIODS
        self.__warmup_periods = cfg.WARMUP_PERIODS
        self.__weights_dir = cfg.WEIGHTS_DIR
        self.__weights_init = cfg.WEIGHTS_INIT
        self.__time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'train', self.__time)

        self.__train_data = Data('train')
        self.__test_data = Data('test')
        self.__steps_per_period = len(self.__train_data)

        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.__label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.__label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.__sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.__mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.__lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.__training = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('learning_rate'):
            self.__global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.__warmup_periods * self.__steps_per_period, dtype=tf.float64,
                                       name='warmup_steps')
            train_steps = tf.constant(self.__max_periods * self.__steps_per_period, dtype=tf.float64,
                                      name='train_steps')
            self.__learn_rate = tf.cond(
                pred=self.__global_step < warmup_steps,
                true_fn=lambda: self.__global_step / warmup_steps * self.__learn_rate_init,
                false_fn=lambda: self.__learn_rate_end + 0.5 * (self.__learn_rate_init - self.__learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.__global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.__global_step, 1.0)

        yolo = YOLOV3(self.__training)
        conv_sbbox, conv_mbbox, conv_lbbox, \
        pred_sbbox, pred_mbbox, pred_lbbox = yolo.build_nework(self.__input_data)

        self.__loss = yolo.loss(conv_sbbox, conv_mbbox, conv_lbbox,
                                pred_sbbox, pred_mbbox, pred_lbbox,
                                self.__label_sbbox, self.__label_mbbox, self.__label_lbbox,
                                self.__sbboxes, self.__mbboxes, self.__lbboxes)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.__learn_rate). \
                minimize(self.__loss, var_list=tf.trainable_variables())
            with tf.control_dependencies([optimizer, global_step_update]):
                self.__train_op = tf.no_op()

        with tf.name_scope('load_save'):
            net_vars = tf.get_collection('yolov3')
            restore_dict = {}
            for var in net_vars:
                var_name = str(var.op.name)
                splited_name = var_name.split('/')
                if splited_name[1] == 'MobilenetV2':
                    org_name = '/'.join(splited_name[1:])
                    restore_dict[org_name] = var
                    logging.info('\t' + str(var_name))
                    logging.info('\t' + str(org_name))
            self.__load = tf.train.Saver(restore_dict) # if want train from a yolo3 ckpt, change restore_dict to net_vars
            self.__save = tf.train.Saver(net_vars, max_to_keep=self.__max_periods)

        with tf.name_scope('summary'):
            self.__loss_ave = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('loss_ave', self.__loss_ave)
            tf.summary.scalar('learn_rate', self.__learn_rate)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())

        self.__sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.__sess.run(tf.global_variables_initializer())
        logging.info('Restoring weights from:\t %s' % self.__weights_init)
        self.__load.restore(self.__sess, self.__weights_init)

    def train(self):
        for period in range(self.__max_periods):
        #     if period == self.__periods_for_step0:
        #         logging.info('Train all of weights')

            print_loss_iter = self.__steps_per_period / 10
            total_train_loss = 0.0
            for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                batch_sbboxes, batch_mbboxes, batch_lbboxes \
                    in self.__train_data:
                _, loss_val, global_step_val = self.__sess.run(
                    [self.__train_op, self.__loss, self.__global_step],
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_sbbox: batch_label_sbbox,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__sbboxes: batch_sbboxes,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: True
                    }
                )
                if np.isnan(loss_val):
                    raise ArithmeticError('The gradient is exploded')
                total_train_loss += loss_val

                if int(global_step_val) % print_loss_iter != 0:
                    continue

                train_loss = total_train_loss / print_loss_iter
                print("->epoch: %d ->steps: %d ->loss: %.4f" % (period, global_step_val, train_loss))
                total_train_loss = 0.0

                self.__sess.run(tf.assign(self.__loss_ave, train_loss))
                summary_val = self.__sess.run(self.__summary_op)
                self.__summary_writer.add_summary(summary_val, global_step_val)
                logging.info('Period:\t%d\tstep:\t%d\ttrain_loss:\t%.4f' % (period, global_step_val, train_loss))

            total_test_loss = 0.0
            for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                batch_sbboxes, batch_mbboxes, batch_lbboxes \
                    in self.__test_data:
                loss_value = self.__sess.run(
                    self.__loss,
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_sbbox: batch_label_sbbox,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__sbboxes: batch_sbboxes,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: False
                    }
                )

                total_test_loss += loss_value
            test_loss = total_test_loss / len(self.__test_data)
            print("test-----loss: %.4f" % test_loss)
            logging.info('Period:\t%d\ttest_loss:\t%.4f' % (period, test_loss))
            saved_model_name = os.path.join(self.__weights_dir, 'yolo.ckpt-%d-%.4f' % (period, test_loss))
            self.__save.save(self.__sess, saved_model_name)
            logging.info('Saved model:\t%s' % saved_model_name)
        self.__summary_writer.close()

if __name__ == '__main__':
    if not os.path.exists(os.path.join(cfg.LOG_DIR, 'train')):
        os.mkdir(os.path.join(cfg.LOG_DIR, 'train'))
    if not os.path.exists(os.path.join(cfg.LOG_DIR, 'test')):
        os.mkdir(os.path.join(cfg.LOG_DIR, 'test'))

    log_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logging.basicConfig(filename='log/train/' + log_time + '.log', format='%(filename)s %(asctime)s\t%(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    logging.info('Batch size for step is:\t%d' % cfg.BATCH_SIZE)
    logging.info('Initial learn rate is:\t%f' % cfg.LEARN_RATE_INIT)
    YoloTrain().train()