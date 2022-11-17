import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
# from tensorflow.compat.v1.nn.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import numpy as np

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
     data_type='fp32', use_negsampling = False, synthetic_input = False, batch_size = 32,
     max_length=100, device = 'gpu'):
        self.synthetic_input = synthetic_input
        self.seq_len_ph = np.ones((batch_size)) * max_length

        if data_type == 'fp32':
            self.model_dtype = tf.float32
            self.model_vdtype = tf.float32
        elif data_type == 'fp16':
            self.model_dtype = tf.float16
            self.model_vdtype = tf.float16
        elif data_type == 'bf16' or data_type == 'bfloat16' :
            self.model_dtype = tf.bfloat16
            self.model_vdtype = tf.float32
        else:
            raise ValueError("Invalid model data type: %s" % data_type)

        if synthetic_input:
            with tf.device('/cpu:0'):
                self.mid_his_batch_ph = tf.random.uniform([batch_size, max_length], 
                    minval = 0, 
                    maxval= n_mid,
                    dtype = tf.int32,
                    name='mid_his_batch_ph')
                self.cat_his_batch_ph = tf.random.uniform([batch_size, max_length], 
                    minval = 0, 
                    maxval= n_cat,
                    dtype = tf.int32,
                    name='cat_his_batch_ph')
                self.uid_batch_ph = tf.random.uniform([batch_size,], 
                    minval = 0, 
                    maxval= n_uid,
                    dtype = tf.int32,
                    name='cat_his_batch_ph')
                self.mid_batch_ph = tf.random.uniform([batch_size,], 
                    minval = 0, 
                    maxval= n_mid,
                    dtype = tf.int32,
                    name='mid_batch_ph') 
                self.cat_batch_ph = tf.random.uniform([batch_size,], 
                    minval = 0, 
                    maxval= n_cat,
                    dtype = tf.int32,
                    name='cat_batch_ph')   
                self.mask = tf.random.uniform([batch_size, max_length], 
                    minval = 0, 
                    maxval= 1, # TODO
                    dtype = self.model_vdtype,
                    name='mask')    
                self.target_ph = tf.random.uniform([batch_size, 2], 
                    minval = 0, 
                    maxval= 1,
                    dtype = self.model_vdtype,
                    name='target_ph') 
                
                self.lr = 0.5 # half it every iteration
                self.use_negsampling = use_negsampling
                if use_negsampling:
                    self.noclk_mid_batch_ph = tf.random.uniform([batch_size, max_length, 5], #TODO, 5 is neg_samples, preset in data_iterator.py
                        minval = 0, 
                        maxval= n_mid,
                        dtype = tf.int32,
                        name='noclk_mid_batch_ph') 
                    self.noclk_cat_batch_ph = tf.random.uniform([batch_size, max_length, 5], 
                        minval = 0, 
                        maxval= n_cat,
                        dtype = tf.int32,
                        name='noclk_cat_batch_ph') 
        else:
            with tf.name_scope('Inputs'):
                self.mid_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
                self.cat_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
                self.uid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='uid_batch_ph')
                self.mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='mid_batch_ph')
                self.cat_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='cat_batch_ph')
                self.mask = tf.compat.v1.placeholder(self.model_vdtype, [None, None], name='mask')
                self.seq_len_ph = tf.compat.v1.placeholder(tf.int32, [None], name='seq_len_ph')
                self.target_ph = tf.compat.v1.placeholder(self.model_vdtype, [None, None], name='target_ph')
                self.lr = tf.compat.v1.placeholder(tf.float64, [])
                self.use_negsampling =use_negsampling
                if use_negsampling:
                    self.noclk_mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph') #generate 3 item IDs from negative sampling.
                    self.noclk_cat_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            if device == 'cpu':
                with tf.device('/cpu:0'):
                    print('embedding on ' + device)
                    self.uid_embeddings_var = tf.compat.v1.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM], dtype=self.model_vdtype)
                    tf.compat.v1.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
                    self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

                    self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], dtype=self.model_vdtype)
                    tf.compat.v1.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
                    self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
                    self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
                    if self.use_negsampling:
                        self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

                    self.cat_embeddings_var = tf.compat.v1.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM], dtype=self.model_vdtype)
                    tf.compat.v1.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
                    self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
                    self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
                    if self.use_negsampling:
                        self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)
            else:
                print('embedding on ' + device)
                self.uid_embeddings_var = tf.compat.v1.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM], dtype=self.model_vdtype)
                tf.compat.v1.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
                self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

                self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM], dtype=self.model_vdtype)
                tf.compat.v1.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
                self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
                self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
                if self.use_negsampling:
                    self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

                self.cat_embeddings_var = tf.compat.v1.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM], dtype=self.model_vdtype)
                tf.compat.v1.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
                self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
                self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
                if self.use_negsampling:
                    self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        #import pdb; pdb.set_trace()
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)# 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])# cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def _sparse_to_dense_grads(self, grads_and_vars):
        return [(tf.convert_to_tensor(g), v) for g, v in grads_and_vars]

    def build_fcn_net(self, inp, use_dice = False):
        def dtype_getter(getter, name, dtype=self.model_dtype, *args, **kwargs):
            var = getter(name, dtype=dtype, *args, **kwargs)
            return var

        def dtype_getter_v(getter, name, dtype=self.model_vdtype, *args, **kwargs):
            var = getter(name, dtype=dtype, *args, **kwargs)
            return var

        with tf.compat.v1.variable_scope("fcn1", custom_getter=dtype_getter, dtype=self.model_dtype):
          with tf.compat.v1.tpu.bfloat16_scope():
            bn1 = tf.compat.v1.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.compat.v1.layers.dense(bn1, 200, activation=None, name='f1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1', data_type=self.model_dtype)
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.compat.v1.layers.dense(dnn1, 80, activation=None, name='f2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2', data_type=self.model_dtype)
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.compat.v1.layers.dense(dnn2, 2, activation=None, name='f3')
          dnn3 = tf.cast(dnn3, tf.float32)
          self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.compat.v1.variable_scope("fcn2", custom_getter=dtype_getter_v, dtype=self.model_vdtype):
          with tf.name_scope("Metrics"):
            #with tf.compat.v1.variable_scope("Metrics", custom_getter=dtype_getter, dtype=self.model_dtype):
                # Cross-entropy loss and optimizer initialization
                ctr_loss = - tf.reduce_mean(tf.math.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                if self.use_negsampling:
                    self.loss += tf.cast(self.aux_loss, self.model_vdtype)
                tf.compat.v1.summary.scalar('loss', self.loss)
                # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
                # self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
                # self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss)

                # convert sparse optimizer to dense optimizer
                adam_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
                gradients = adam_optimizer.compute_gradients(self.loss)
                gradients = self._sparse_to_dense_grads(gradients)
                self.optimizer = adam_optimizer.apply_gradients(gradients)

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), self.model_vdtype))
                tf.compat.v1.summary.scalar('accuracy', self.accuracy)

          self.merged =  tf.compat.v1.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        def dtype_getter(getter, name, dtype=self.model_dtype, *args, **kwargs):
            var = getter(name, dtype=dtype, *args, **kwargs)
            return var

        with tf.compat.v1.variable_scope("aux_loss", custom_getter=dtype_getter, dtype=self.model_dtype):
            click_input_ = tf.concat([h_states, click_seq], -1)
            noclick_input_ = tf.concat([h_states, noclick_seq], -1)
            click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
            noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
            click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
            noclick_loss_ = - tf.reshape(tf.math.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
            loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
            return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        def dtype_getter(getter, name, dtype=self.model_dtype, *args, **kwargs):
            var = getter(name, dtype=dtype, *args, **kwargs)
            return var

        with tf.compat.v1.variable_scope("aux_net", custom_getter=dtype_getter, dtype=self.model_dtype):
            bn1 = tf.compat.v1.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.compat.v1.AUTO_REUSE)
            dnn1 = tf.compat.v1.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.compat.v1.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.compat.v1.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.compat.v1.AUTO_REUSE)
            dnn2 = tf.nn.sigmoid(dnn2)
            dnn3 = tf.compat.v1.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.compat.v1.AUTO_REUSE)
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
            return y_hat

    def train_synthetic_input(self, sess):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer])
        return loss, accuracy, 0

    def train(self, sess, inps, timeline_flag=False, options=None,run_metadata=None, step=None):
        if self.use_negsampling:
            if timeline_flag:
                loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], 
                    options=options,run_metadata=run_metadata,
                    feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.lr: inps[8],
                    self.noclk_mid_batch_ph: inps[9],
                    self.noclk_cat_batch_ph: inps[10],
                })

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()

                with open('./timeline/dien_timeline_bfloat16.json', 'w') as f:
                    f.write(chrome_trace)
            else:
                loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.lr: inps[8],
                    self.noclk_mid_batch_ph: inps[9],
                    self.noclk_cat_batch_ph: inps[10],
                })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps, timeline=False, options=None,run_metadata=None):
        if self.use_negsampling:
            if timeline:
                probs, loss, accuracy, aux_loss = sess.run(
                    [self.y_hat, self.loss, self.accuracy, self.aux_loss], options=options,run_metadata=run_metadata, 
                    feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.noclk_mid_batch_ph: inps[8],
                    self.noclk_cat_batch_ph: inps[9],
                })
            else:
                probs, loss, accuracy, aux_loss = sess.run(
                    [self.y_hat, self.loss, self.accuracy, self.aux_loss], 
                    feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.noclk_mid_batch_ph: inps[8],
                    self.noclk_cat_batch_ph: inps[9],
                })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DIN_V2_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.compat.v1.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.compat.v1.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=att_outputs,
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.compat.v1.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)

class Model_DIN_V2_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.compat.v1.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.compat.v1.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs2, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.compat.v1.summary.histogram('att_fea', att_fea)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # Fully connected layer
        bn1 = tf.compat.v1.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.compat.v1.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.compat.v1.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.compat.v1.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_eb,self.item_his_eb_sum], axis=-1),
                                self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.compat.v1.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.math.log(self.y_hat) * self.target_ph)
            tf.compat.v1.summary.scalar('loss', self.loss)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.compat.v1.summary.scalar('accuracy', self.accuracy)
        self.merged =  tf.compat.v1.summary.merge_all()


class Model_DIN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                         EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                         use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.compat.v1.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.compat.v1.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.compat.v1.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type='fp32', use_negsampling=False, 
     synthetic_input = False, batch_size = 32, max_length=100, device = 'gpu'):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE, data_type,
                                                          use_negsampling, synthetic_input, batch_size, max_length, device)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)

class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.compat.v1.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type='fp32', 
        use_negsampling=True, synthetic_input = False, batch_size = 32, max_length=100, device = 'gpu'):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, data_type,
                                                          use_negsampling, synthetic_input, batch_size, max_length, device)
        def castif(inp):
          return tf.cast(inp, self.model_dtype) if inp.dtype != self.model_dtype else inp

        def castb(inp):
          return tf.cast(inp, self.model_vdtype) if inp.dtype != self.model_vdtype else inp

        def dtype_getter(getter, name, dtype=self.model_dtype, *args, **kwargs):
            var = getter(name, dtype=dtype, *args, **kwargs)
            return var

        with tf.compat.v1.variable_scope("dien", custom_getter=dtype_getter, dtype=self.model_vdtype):
            # RNN layer(-s)
            with tf.name_scope('rnn_1'):
              with tf.compat.v1.tpu.bfloat16_scope():
                rnn_outputs, _ = dynamic_rnn(tf.compat.v1.nn.rnn_cell.GRUCell(HIDDEN_SIZE), inputs=castif(self.item_his_eb),
                                             sequence_length=self.seq_len_ph, dtype=self.model_dtype,
                                             scope="gru1")
                tf.compat.v1.summary.histogram('GRU_outputs', rnn_outputs)

            aux_loss_1 = self.auxiliary_loss(castb(rnn_outputs[:, :-1, :]), self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(self.item_eb, castb(rnn_outputs), ATTENTION_SIZE, self.mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
                tf.compat.v1.summary.histogram('alpha_outputs', alphas)

            with tf.name_scope('rnn_2'):
              with tf.compat.v1.tpu.bfloat16_scope():
                rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                         att_scores = tf.expand_dims(castif(alphas), -1),
                                                         sequence_length=self.seq_len_ph, dtype=self.model_dtype,
                                                         scope="gru2")
                tf.compat.v1.summary.histogram('GRU2_Final_State', final_state2)

            final_state2 = tf.cast(final_state2, self.model_vdtype)
            inp = castif(tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1))
            self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
     synthetic_input = False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling, synthetic_input)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.compat.v1.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.compat.v1.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.compat.v1.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)
