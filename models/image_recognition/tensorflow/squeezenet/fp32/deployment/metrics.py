import tensorflow as tf


class Metrics(object):
    def __init__(self, labels, clone_logits, clone_predictions,  device, name,
                 padded_data=False):
        self.labels = labels
        self.clone_predictions = clone_predictions
        self.clone_logits = clone_logits
        self.device = device
        self.name = name
        self.padded_data = padded_data
        self.accuracy = None
        self.update_op = None
        self.reset_op = None
        self._generate_metrics()

    def _generate_metrics(self):
        with tf.variable_scope('metrics'), tf.device(self.device):
            with tf.variable_scope(self.name):
                predictions = tf.concat(
                    values=self.clone_predictions,
                    axis=0
                )
                logits = tf.concat(
                    values=self.clone_logits,
                    axis=0
                )

                if self.padded_data:
                    not_padded = tf.not_equal(self.labels, -1)
                    self.labels = tf.boolean_mask(self.labels, not_padded)
                    predictions = tf.boolean_mask(predictions, not_padded)
                    logits = tf.boolean_mask(logits, not_padded)
                
                self.accuracy, self.update_op = tf.metrics.accuracy(
                    labels=self.labels,
                    predictions=predictions
                )

                """               
                self.accuracy, self.update_op = tf.metrics.recall_at_k(
                    self.labels,tf.squeeze(logits,[2]),5)
                    
                self.accuracy, self.update_op = tf.metrics.recall_at_top_k(
                    labels=self.labels,
                    predictions_idx=predictions,
                    k=1,
                )
                """

                accuracy_vars = tf.contrib.framework.get_local_variables(
                    scope='metrics/{}/accuracy'.format(self.name)
                )
                self.reset_op = tf.variables_initializer(
                    var_list=accuracy_vars
                )
