import tensorflow as tf 
import os
from time import time
import numpy as np
keras = tf.keras

#------------------------UTILS-------------------
def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def gen_conv(batch_input, out_channels, strides=1):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=strides, padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=3, strides=(2, 2), padding="same", kernel_initializer=initializer)

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def conv_bn_relu(x, filters, training):
    conv = gen_conv(x, filters)
    bn = batchnorm(conv, training)
    return tf.nn.relu(bn)

def down_block(input, ngf,  training, pool_size):
    x = tf.layers.max_pooling2d(input, 2, 2)
    temp = conv_bn_relu(x, ngf, training)
    bn = batchnorm(gen_conv(temp, ngf), training)
    bn += x
    if pool_size == 4:
        bn = tf.layers.max_pooling2d(bn, 2, 2)
    act = tf.nn.relu(bn)
    return bn, act

def tf_reshape_tensor(inputs, tf_target_shape):
    '''
        params
            inputs: tensorflow input tensor
            tf_target_shape: a tf.shape object
        return:
            a tensor with targeted shape
    '''
    h, w = tf_target_shape[1], tf_target_shape[2]
    tensor_reshaped = tf.image.resize_images(
        inputs,
        (h, w),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=False
    )
    return tensor_reshaped

def up_block(act, bn, ngf, use_drop,training, concat=True):
    x = tf_reshape_tensor(act, tf.shape(bn))
    if concat:
        temp = tf.concat([bn, x], axis=-1)
    else:
        temp = x
    temp = conv_bn_relu(temp, ngf, training)
    bn = batchnorm(gen_conv(temp, ngf), training)
    output = tf.nn.relu(bn)
    if use_drop:
        output = tf.nn.dropout(output, keep_prob=0.5)
    return output


#--------------------------------GENERATOR
# def get_generator_deepunet(generator_inputs, generator_outputs_channels, 
# ngf, use_drop=True, training=True):
def get_generator_deepunet(generator_inputs, generator_outputs_channels, ngf, class_name, use_drop=True,
                           training=True, verbal=True):
    '''
        generator_inputs:512x512x3
        outputs: for line and for text
    '''

    assert generator_outputs_channels is not None
    x = conv_bn_relu(generator_inputs, 64, training)
    net = conv_bn_relu(x, ngf, training)
    bn1 = batchnorm(gen_conv(net, ngf), training=training)
    act1 = tf.nn.relu(bn1)
    bn2, act2 = down_block(act1, ngf, pool_size=4, training=training)
    bn3, act3 = down_block(act2, ngf, pool_size=4, training=training)
    bn4, act4 = down_block(act3, ngf, pool_size=2, training=training)
    bn5, act5 = down_block(act4, ngf, pool_size=2, training=training)
    bn6, act6 = down_block(act5, ngf, pool_size=2, training=training)
    bn7, act7 = down_block(act6, ngf, pool_size=2, training=training)
    temp1 = up_block(act6, bn7, ngf, use_drop=use_drop, training=training)
    temp2 = up_block(temp1, bn6, ngf, use_drop=use_drop, training=training)
    temp3 = up_block(temp2, bn5, ngf, use_drop=use_drop, training=training)
    temp4 = up_block(temp3, bn4, ngf, use_drop=use_drop, training=training)
    temp5 = up_block(temp4, bn3, ngf, use_drop=use_drop, training=training)
    temp6 = up_block(temp5, bn2, ngf, use_drop=use_drop, training=training)
    temp7 = up_block(temp6, bn1, ngf, use_drop=use_drop, training=training)
    with tf.variable_scope('encode_output_{}'.format(class_name)):
        logits = gen_conv(temp7, generator_outputs_channels)

    layers = [x, act1, act2, act3, act4, act5, act6, act7,
              temp1, temp2, temp3, temp4, temp5, temp6, temp7, logits]
    if verbal:
        for l in layers:
            print(l.shape)
    return logits



def get_generator_unet(generator_inputs, generator_outputs_channels, ngf, use_drop=True, \
    training=True, verbal=True):
    '''
        generator_inputs:512x512x3
        outputs: for line and for text
    '''
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    scope_name = "encoder_1"
    with tf.variable_scope(scope_name):
        output = gen_conv(generator_inputs, ngf, strides=2)
        layers.append(output)
    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        scope_name = "encoder_%d" % (len(layers) + 1)
        with tf.variable_scope(scope_name):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, strides=2)
            output = batchnorm(convolved, training)
            layers.append(output)
    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        scope_name = "decoder_%d" % (skip_layer + 1)
        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output, training=training)

            if dropout > 0.0 and use_drop:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
                
            layers.append(output)
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    scope_name = "outputs"
    with tf.variable_scope(scope_name):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        logits = gen_deconv(rectified, generator_outputs_channels)
        layers.append(logits)
    if verbal:
        print(generator_inputs.shape)
        for layer in layers:
            print(layer.shape)
    return layers[-1]



def load_image(path):
    '''
    inputs must be converted to png file
    '''
    assert path[-3:] == 'png'
    raw = tf.read_file(path)
    #raw -> image(range[0,255])
    img_png = tf.image.decode_png(raw)
    #range[0, 255]-> range[0, 1]
    img_float = tf.image.convert_image_dtype(img_png, dtype=tf.float32)
    return img_float

def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)

def export_generator(fn_generator, ngf, stride, crop_size, checkpoint, output_dir):
    CROP_SIZE = crop_size
    strides = [stride, stride]
    preprocess = lambda x: x*2-1
    deprocess = lambda x: (x+1)/2

    def extract_patches(image, k_size, strides):
        images = tf.extract_image_patches(tf.expand_dims(
            image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
        images_shape = tf.shape(images)
        # print([images_shape[0]*images_shape[1], *k_size[1:3], 3])
        images_reshape = tf.reshape(
            images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
        images, n1, n2 = tf.cast(
            images_reshape, tf.uint8), images_shape[0], images_shape[1]
        return images, n1, n2


    def join_patches(images, n1, n2, k_size, strides):
        s1 = k_size[1]//2-strides[1]//2
        s2 = k_size[2]//2-strides[2]//2
        roi = images[:,
                        s1:s1+strides[1],
                        s2:s2+strides[2],
                        :]
        new_shape = [n1, n2, *roi.get_shape().as_list()[1:]]
        print(new_shape)
        reshaped_roi = tf.reshape(roi, new_shape)
        reshaped_roi = tf.transpose(reshaped_roi, perm=[0, 2, 1, 3, 4])
        rs = tf.shape(reshaped_roi)
        rv = tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1])
        return rv

    def resize(image, new_size=None):
        shape = tf.shape(image)
        h, w = shape[0], shape[1]
        if new_size is None:
            new_h = tf.cast(tf.ceil(h/CROP_SIZE)*CROP_SIZE, tf.int32)
            new_w = tf.cast(tf.ceil(w/CROP_SIZE)*CROP_SIZE, tf.int32)
        else:
            new_h, new_w = new_size
        return tf.image.resize_bilinear(tf.expand_dims(image, 0), (new_h, new_w))[0]
        # inputs = tf.placeholder(tf.float32, [None, *CROP_SIZE, 3], 'inputs')
    inputs = tf.placeholder(tf.float32, [None, None, 3], 'inputs')
    inputs_shape = tf.shape(inputs)
    input_resized = resize(inputs)
    images, n1, n2 = extract_patches(
        input_resized, [1, CROP_SIZE, CROP_SIZE, 1], [1, *strides, 1])
    n1 = tf.identity(n1, 'n1')
    n2 = tf.identity(n2, 'n2')
    batch_input_tensor = tf.identity(images / 255, 'batch_input_tensor')
    batch_input_placeholder = tf.placeholder(
        tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_input_placeholder')
    with tf.variable_scope('generator'):
        logits = fn_generator(preprocess(batch_input_placeholder),2, ngf=ngf)
    ft = tf.concat([logits[...,:1], logits[...,:1], logits[...,-1:]], axis=-1)
    batch_output = deprocess(tf.tanh(ft))
    print('batch_output', batch_output)

    h, w = batch_input_placeholder.get_shape().as_list()[1:3]
    batch_output = tf.image.resize_bilinear(batch_output, (h, w))
    batch_output_tensor = tf.identity(
        batch_output, name='batch_output_tensor')
    batch_output_placeholder = tf.placeholder(
        tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_output_placeholder')
        
    batch_output = join_patches(batch_output_placeholder, n1, n2, [
                                1, CROP_SIZE, CROP_SIZE, 1], [1, *strides, 1])
    batch_output = resize(batch_output, [inputs_shape[0], inputs_shape[1]])
    outputs = tf.identity(
        tf.cast(batch_output*255, tf.uint8), name='outputs')

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        print("loading model from checkpoint", checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("exporting model:", checkpoint)
        export_saver.export_meta_graph(
            filename=os.path.join(output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(
            output_dir, "export"), write_meta_graph=False)
    return


class unet_model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.sess = tf.Session()
        self.build()

    def build(self):
        meta_path = os.path.join(self.checkpoint, 'export.meta')
        tf.train.import_meta_graph(meta_path)
        saver = tf.train.Saver()

        self.inputs = get_tensor_by_name('inputs')
        self.outputs = get_tensor_by_name('outputs')
        self.batch_input_tensor = get_tensor_by_name('batch_input_tensor')
        self.batch_input_placeholder = get_tensor_by_name('batch_input_placeholder')
        self.batch_output_tensor = get_tensor_by_name('batch_output_tensor')
        self.batch_output_placeholder = get_tensor_by_name('batch_output_placeholder')
        self.n1 = get_tensor_by_name('n1')
        self.n2 = get_tensor_by_name('n2')
        saver.restore(self.sess, tf.train.latest_checkpoint(
            self.checkpoint))

    def run(self, feed_image, batch_size=8):
        batch_input, n1_val, n2_val = self.sess.run([self.batch_input_tensor, self.n1, self.n2], {self.inputs:feed_image})
        rv = []
        start = time()
        for i in range(0, len(batch_input), batch_size):
            print('\r {:0.2f} %'.format(i/len(batch_input)), end='')
            rv.append(self.sess.run(self.batch_output_tensor, {self.batch_input_placeholder: batch_input[i:i+batch_size]}))
        output_feed = np.concatenate(rv, axis=0)

        print('Patch speed: ', len(batch_input)/(time()-start))

        return self.sess.run(self.outputs, {self.batch_output_placeholder: output_feed, self.inputs:feed_image})





def keras_unet(generator_inputs,generator_outputs_channels=2, ngf=32, use_drop=True, training=True):
    
    layers = []

    _ = keras.layers.Conv2D(ngf, 3, 2, padding='same')(generator_inputs)
    layers.append(_)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    for out_channels in layer_specs:
        _ = keras.layers.LeakyReLU(.2)(_)
        _ = keras.layers.Conv2D(out_channels, strides=2, kernel_size=3, padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)
        layers.append(_)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1


        if decoder_layer == 0:
            _ = layers[-1]
        else:
            _ = keras.layers.Concatenate()([layers[-1], layers[skip_layer]])#tf.concat([layers[-1], layers[skip_layer]], axis=3)

        _ = keras.layers.Activation('relu')(_)
        _ = keras.layers.Conv2DTranspose(out_channels, strides=2, kernel_size=3, padding='same')(_)
        _ = keras.layers.BatchNormalization()(_)

        if dropout > 0.0 and use_drop:
            _ = keras.layers.Dropout(dropout)(_)
        layers.append(_)
        
    input = keras.layers.Concatenate(axis=-1)([layers[-1], layers[0]])#tf.concat([layers[-1], layers[0]], axis=3)
    rectified = keras.layers.Activation('relu')(input)
    outputs = keras.layers.Conv2DTranspose(generator_outputs_channels, strides=2, kernel_size=3, padding='same', activation='sigmoid')(rectified)
    
    return keras.models.Model(generator_inputs, outputs)
