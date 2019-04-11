import tensorflow as tf
import os
from time import time
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from pyson.unet import get_generator_deepunet
from time import time
import tensorflow as tf
from skimage import measure
import cv2
import json
from pyson.utils import read_img, resize_by_factor, timeit, findContours, show


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __call__(self):
        return (self.x, self.y)


class Line:
    def __init__(self, a, b):
        self.type = self._build(a, b)

    def __call__(self):
        return self.p_min, self.p_max

    def _build(self, a, b):
        y1, x1, y2, x2 = a.y, a.x, b.y, b.x
        dy = abs(y1-y2)
        dx = abs(x1-x2)
        self.line_type = 'v' if dy > dx else 'h'
        if self.line_type == 'v':
            vala, valb = a.y, b.y
        else:
            vala, valb = a.x, b.x
        if vala > valb:
            self.p_max = a
            self.p_min = b
        else:
            self.p_max = b
            self.p_min = a
        self.epsilon = .1*self._length()

    def get_intersect_point(self, line):
        y1, x1, y2, x2 = self.p_min.y, self.p_min.x, self.p_max.y, self.p_max.x
        y3, x3, y4, x4 = line.p_min.y, line.p_min.x, line.p_max.y, line.p_max.x
        px_top = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
        bot = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        py_top = (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)
        return Point(int(px_top/bot), int(py_top/bot))

    def nearestLine(self, point, all_lines, epsilon):
        min_distance = float('inf')
        rt_line = None
        care_lines = self._get_care_lines(point, all_lines)
        for line in care_lines:
            dist = self._distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                rt_line = line
        return rt_line

    def _distancetoline(self, point, line):
        y1, x1, y2, x2 = line.p_min.y, line.p_min.x, line.p_max.y, line.p_max.x
        y0, x0 = point.y, point.x
        # +delta_hinhchieu+delta_vitri
        return abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2 * x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

    def _get_care_lines(self, point, all_lines):
        if point is self.p_min:
            if self.line_type == 'h':  # y
                rv = [line for line in all_lines if line.p_max.x < point.x and point.y <
                      line.p_max.y+self.epsilon and point.y > line.p_min.y-self.epsilon]
            if self.line_type == 'v':  # x
                rv = [line for line in all_lines if line.p_min.y < point.y and point.x <
                      line.p_max.x+self.epsilon and point.x > line.p_min.x-self.epsilon]
        if point is self.p_max:
            if self.line_type == 'h':  # y
                rv = [line for line in all_lines if line.p_min.x > point.x and point.y <
                      line.p_max.y+self.epsilon and point.y > line.p_min.y-self.epsilon]
            if self.line_type == 'v':  # v
                rv = [line for line in all_lines if line.p_min.y > point.y and point.x <
                      line.p_max.x+self.epsilon and point.x > line.p_min.x-self.epsilon]

        return rv

    def _length(self):
        a, b = self()
        xa, ya = a()
        xb, yb = b()
        dx = abs(xa-xb)
        dy = abs(ya-yb)
        return np.sqrt(dx**2+dy**2)


class AnalyseLine:
    '''
        prepare: a model to get a dictionary of {'inputs':, 'outputs':} 
        and a session to run it
    '''

    def __init__(self, input_path,
                 unet_model=None,
                 model_output=None,
                 useNoneIntersect=True,
                 thresh_rect=.6,
                 thresh_model_min=150,
                 epsilon=11,
                 output_dir='__table_analisys__'):

        self.thresh_rect = thresh_rect
        self.thresh_model_min = thresh_model_min
        self.useNoneIntersect = useNoneIntersect
        self.epsilon = epsilon
        self.intersect_points = []
        self.output_dir = output_dir
        if use_model is None:
            assert model_output is not None
            self.model_output = model_output
        else:
            self.unet_model = unet_model
            self.model_output = self.unet_model.predict(input_path)
        self._build()

    def _build(self):
        '''
            From model output decompose it to self.v_lines, and self.h_lines
        '''
        assert self.model_output is not None, 'call function get_model_output to\
         get the model output into the pipeline'

        def f_(img, mode='h', img_2=None):
            assert img_2 is not None, self.useNoneIntersect is not None
            if mode == 'h':
                channel = 1
            elif mode == 'v':
                channel = 0
            else:
                assert mode == 'h' or mode == 'v'

            labels = measure.label(img, neighbors=8, background=0)
            pad = np.zeros_like(img)
            lines = []
            for label in np.unique(labels):
                if label == 0:
                    continue

                mask = labels == label
                mask = (mask * 255).astype('uint8')
                mask_2 = mask * img_2
                non_zeros = np.count_nonzero(mask_2)
                if non_zeros == 0 and self.useNoneIntersect == False:
                    # if this line doesn't intersect with anyother line then ignore it
                    continue
                idxs = np.column_stack(np.where(mask > 0))
                min_idx, max_idx = np.argmin(
                    idxs[:, channel]), np.argmax(idxs[:, channel])
                a, b = idxs[min_idx][::-1] + 1, idxs[max_idx][::-1] + 1
                cv2.line(pad, tuple(a), tuple(b), 255, 3)
                lines.append((a, b))

            return pad, lines
        if len(self.model_output.shape) == 3:
            gray = cv2.cvtColor(self.model_output, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.model_output
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, self.thresh_model_min,
                               255, cv2.THRESH_BINARY_INV)[1]
        ngang = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones([1, 50]))
        doc = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones([50, 1]))
        self.img_h, h_lines = f_(ngang, 'h', doc)
        self.img_v, v_lines = f_(doc, 'v', ngang)
        self.h_lines = [Line(Point(x1, y1), Point(x2, y2))
                        for ((x1, y1), (x2, y2)) in h_lines]
        self.v_lines = [Line(Point(x1, y1), Point(x2, y2))
                        for ((x1, y1), (x2, y2)) in v_lines]

        self.processed = ngang+doc
        self.display = None

    def show(self):
        show(self.display, size=5)

    def findConvexhull(self):
        assert self.processed is not None

        def is_rectangle(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(
                cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if h*w > 0 and len(approx) == 2:
                rate = cv2.contourArea(cnt) / (h*w)
                return rate > self.thresh_rect
            else:
                return False

        mask = np.zeros_like(self.img_inp.copy())
        cnts, hiers = findContours(self.processed)
        for cnt in cnts:
            cv_cnt = cv2.convexHull(cnt)
            if is_rectangle(cv_cnt):
                x, y, w, h = cv2.boundingRect(cnt)
                approx = cv2.approxPolyDP(
                    cnt, 0.08 * cv2.arcLength(cnt, True), True)
                mask = putText(mask, (x, y), str(len(approx)))
                cv2.drawContours(mask, [cv_cnt], 0, 255, -1)
        self.display = mask
        self.processed = mask

    def merge_input(self):

        mask = self.display
        if len(mask.shape) == 2:
            z = np.zeros_like(mask)
            mask = np.stack([mask, z, z], axis=-1)
            mask = np.array(mask, dtype='uint8')
        inp = 255-self.img_inp
        if len(inp.shape) == 2:
            inp = np.stack([inp]*3, axis=-1)
        inp = 0.2*inp + 0.8*mask
        self.display = inp.astype('uint8')

    def mask2cells(self):
        def is_rect(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 0 and w*h > 0 and area / w*h > 0.6:
                return True
            else:
                return False

        def get_num_cell(hiers):
            x = hiers[:, -1]
            unique, counts = np.unique(x, return_counts=True)
            return unique[1:]

        cnts, hiers = findContours(self.processed)
        mask_debug = np.zeros([*self.processed.shape[:2], 3], dtype='uint8')
        i = 0
        j = 1
        k = 2
        out_cnts = {}
        for ci, (cnt, h) in enumerate(zip(cnts, hiers)):
            if is_rect(cnt):
                color1 = int((i % 10*255/9))+1
                color2 = int((j % 10*255/9))+1
                color3 = int((k % 10*255/9))+1
                i += np.random.choice(10)
                j += np.random.choice(10)
                k += np.random.choice(10)
                cv2.drawContours(
                    mask_debug, [cnt], 0, (color1, color2, color3), -1)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                cnt = np.int0(box)
                if h[-1] == -1:
                    out_cnts['table_{}'.format(ci)] = cnt
                else:
                    out_cnts['cell_{}_table_{}'.format(ci, h[-1])] = cnt

        self.cnts, self.display = out_cnts, mask_debug

    def all_intersect_points(self):
        def is_close(p1, p2):
            y1, x1 = p1
            y2, x2 = p2
            dx = abs(x1-x2)
            dy = abs(y1-y2)
            l = np.sqrt(dx**2+dy**2)
            return l < self.epsilon

        mask = np.zeros_like(self.display)
        for p in self.intersect_points:
            cv2.circle(mask, p(), 5, (0, 255, 0), -1)
        for line in self.h_lines+self.v_lines:
            a_, b_ = line()
            cv2.line(mask, a_(), b_(), (255, 0, 255), 3)

        self.display = mask

    def adjust(self):
        '''
        This process return h_lines, and v_lines
        '''
        def f_(h_lines, v_lines):
            '''
                mask: only for debug
            '''
            rv = []
            for h_idx, h_line in enumerate(h_lines):
                v_line1 = h_line.nearestLine(
                    h_line.p_min, v_lines, self.epsilon)
                v_line2 = h_line.nearestLine(
                    h_line.p_max, v_lines, self.epsilon)
                x = h_line
                if v_line1 is not None:
                    intersec_point1 = h_line.get_intersect_point(v_line1)
                    x = Line(intersec_point1, x.p_max)
                if v_line2 is not None:
                    intersec_point2 = h_line.get_intersect_point(v_line2)
                    x = Line(x.p_min, intersec_point2)
                    self.intersect_points.append(intersec_point2)
                rv.append(x)
            return rv

        h, w = self.model_output.shape[:2]
        mask = np.zeros([h, w, 3], dtype='uint8')

        self.h_lines = f_(self.h_lines, self.v_lines)
        self.v_lines = f_(self.v_lines, self.h_lines)

        for line in self.h_lines+self.v_lines:
            a_, b_ = line()
            cv2.line(mask, a_(), b_(), (255, 0, 255), 3)
        self.processed = 255-mask[:, :, 0]
        self.display = mask

    def dump_output(self, save=True):
        assert self.cnts is not None, self.cnts
        os.makedirs(self.output_dir, exist_ok=True)
        json_save_path = os.path.join(self.output_dir, self.image_name+'.json')
        image_save_path = os.path.join(self.output_dir, self.image_name+'.png')
        for k, v in self.cnts.items():
            self.cnts[k] = str(v)
        if save:
            json.dump(self.cnts, open(json_save_path, 'w'))
            write_img(image_save_path, self.display)
            print('Image output:', image_save_path)
            print('Json output:', json_save_path)

        return self.cnts, self.display


class UnetModel():
    def __init__(self, class_name, checkpoint='models_checkpoint'):
        self.class_name = class_name
        self.max_height_width = 2048
        self._build(checkpoint=checkpoint)

    def _build(self, checkpoint):
        inputs = tf.placeholder(tf.float32, [None, None, None, 1])
        self.model = self._fn_build_model(inputs, ngf=32)
        saver = tf.train.Saver(max_to_keep=5)
        self.sess = tf.Session()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver.restore(self.sess, latest_checkpoint)
        self.model, self.sess

    def _fn_build_model(self, inputs, targets=None, ngf=16,
                        weight_loss=1, optimizer=tf.train.AdamOptimizer, image_scale=[0, 1]):
        '''
            params:
        '''
        with tf.name_scope('scope_{}'.format(self.class_name)):
            with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
                logits = get_generator_deepunet(inputs, 1, ngf,
                                                '{}_logits'.format(
                                                    self.class_name),
                                                verbal=False,
                                                use_drop=False)
                outputs = tf.sigmoid(logits)
        return {'inputs': inputs, 'outputs': outputs}

    @timeit
    def predict(self, path):
        img_inp = read_img(path, True, False)
        h, w = img_inp.shape[:2]
        f = max(img_inp.shape[:2]) / self.max_height_width
        if f > 1:
            img_inp = resize_by_factor(img_inp, 1/f)
        img = np.expand_dims(img_inp, 0)

        img = np.expand_dims(img, -1)/255.0
        output = self.sess.run(self.model['outputs'][0, ..., 0], {
                               self.model['inputs']: img})
        model_output = (output*255).astype('uint8')
        return cv2.resize(model_output,(w, h))

    def __str__(self):
        return 'Model: {}, Inputs: {}, Output: {}'.format(self.class_name, self.model['inputs'].shape, self.model['outputs'].shape)


if __name__ == '__main__':
    img_path = '/Users/bi/Downloads/pdftoimage/Usen_invoice_1/Usen_invoice_1-3.jpg'
    unet_model = UnetModel('border', 'models_checkpoint')
    model_analizer = AnalyseLine(img_path, unet_model)
    model_analizer.adjust()
    model_analizer.mask2cells()
    model_analizer.show()
