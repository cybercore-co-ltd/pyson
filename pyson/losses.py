import tensorflow as tf
from tensorflow.python.keras import backend as K


def ctc_loss(labels, y_pred):
    '''
        Arguments:
            labels: arrays with shape [batch_size, time_steps]  
                            Each time step is an interger,
            y_pred: arrays with shape [batch_size, time_steps, num_of_class] 
                            Each time step is a numpy array with length of num_of_class
        Return:
            loss: single float number # ex: 0.12312
    '''
    def caculate_y_pred_lengths(y_pred):    
        x = tf.shape(y_pred)[1]
        x = tf.tile([x], tf.shape(y_pred)[:1])
        return tf.reshape(tf.squeeze(x), [-1, 1])    
    
    def caculate_label_lengths(labels, padding_value=-1):
        i0 = tf.constant(0)
        m0 = tf.constant([1], shape=[1,1])
        c = lambda i, m: i < tf.shape(labels)[0]
        def b(i, m):
            label_i = labels[i]
            invalid_indexes = tf.reshape(tf.where(tf.equal(label_i, padding_value)), [-1])        
            l = tf.shape(invalid_indexes)[0]
            li = tf.cond(tf.equal(l, 0), 
                         lambda: tf.shape(label_i)[0], 
                         lambda: tf.cast(invalid_indexes[0], tf.int32)
                        )
            li = tf.reshape(li, [1,1])
            m = tf.concat([m, li], axis=0)
            m = tf.reshape(m, [-1, 1])
            return [i+1, m]


        out_loop = tf.while_loop(
            c, b, loop_vars=[i0, m0],
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, 1])])
        out = out_loop[1][1:]
        out = tf.cast(out, tf.int32)
        return out
    
    labels = tf.concat([labels,tf.ones(shape=[tf.shape(labels)[0], 1])*-1], axis=-1)
    input_length = caculate_y_pred_lengths(y_pred)
    label_length = caculate_label_lengths(labels)
    label_length = tf.reshape(label_length, [-1, 1])
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
                    
    
def ctc_loss_v2(labels, y_pred):
    '''
        Arguments:
            labels: arrays with shape [batch_size, time_steps]  
                            Each time step is an interger,
            y_pred: arrays with shape [batch_size, time_steps, num_of_class] 
                            Each time step is a numpy array with length of num_of_class
        Return:
            loss: single float number # ex: 0.12312
    '''
    def caculate_y_pred_lengths(y_pred):    
        x = tf.shape(y_pred)[1]
        x = tf.tile([x], tf.shape(y_pred)[:1])
        x = tf.reshape(tf.squeeze(x), [-1, 1])    
        x = tf.identity(x, 'y_pred_length')
        return x
    
    def caculate_label_lengths(labels, padding_value=-1):
        x = labels
        x = tf.logical_not(tf.equal(x, padding_value))
        x = tf.cast(x, tf.int32)
        x = tf.reduce_sum(x, -1)
        x =  tf.reshape(x, [-1, 1])
        x = tf.identity(x, 'label_length')
        return  x 
    
    labels = tf.identity(labels, 'labels')
    y_pred = tf.identity(y_pred, 'y_preds')
    input_length = caculate_y_pred_lengths(y_pred)
    
    label_length = caculate_label_lengths(labels)
#     for x in [labels, y_pred, input_length, label_length]: print(x.name, x.shape)
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
                                           
                                           
def suppervise_ctc_loss(labels, preds , strides):
    i_batch = tf.constant(0)
    loss_batch = tf.constant(0, tf.float32, shape=[])
    batch_size_tensor = tf.shape(strides)[0]
    cond = lambda i_batch, loss: tf.less(i_batch, batch_size_tensor)

    def batch_body(i_batch, loss_batch):
        stride = strides[i_batch]
        pred = preds[i_batch]
        label = labels[i_batch]
        i = tf.constant(0)
        loss = tf.constant(0, tf.float32, shape=[])
        to_be_continue = True
        def text_line_body(i, loss):
            x2_start, x2_end = stride[i][0], stride[i][1]
            def character_ctc_loss():
                _lbl = label[i: i+1]
                _pred = pred[x2_start:x2_end]
                _current_loss = ctc_loss(_lbl, [_pred]) / tf.cast((x2_end-x2_start), tf.float32)
                return loss + _current_loss
            loss = tf.cond(tf.equal(x2_start, -1), lambda: loss,  character_ctc_loss)
            return i+1, loss

        max_idx = tf.reduce_max(tf.where(tf.not_equal(labels[i_batch,:,0], -1)))
        max_idx = tf.cast(max_idx, tf.int32)

        cond = lambda i, loss: tf.less(i, max_idx)
        _, loss_out = tf.while_loop(cond, text_line_body, loop_vars=[i, loss])
        loss_out = loss_out+loss_batch
        return i_batch+1, loss_out

    out_loop = tf.while_loop(cond, batch_body, loop_vars=[i_batch, loss_batch])[-1]
    return out_loop
