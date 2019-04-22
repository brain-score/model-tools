class TFProbabilitiesClassifier:
    def __init__(self,
                 init_lr=1e-4,
                 max_epochs=40, 
                 batch_size=64, activation=None,
                 fc_weight_decay=1e3,
                 fc_dropout = 1.0,
                 tol=1e-4,
                 log_rate=5, gpu_options=None):
        """
        mapping function class.
        :param batch_size: batch size
        :param activation: what activation to use if any
        :param init_lr: initial learning rate
        :param tol: tolerance - stops the optimization if reaches below tol
        :param fc_weight_decay: regularization coefficient for fully connected layers (inverse of sklearn C)
        :params fc_dropout: dropout parameter for fc layers
        :param log_rate: rate of logging the loss values (in epochs)
        """
        self._batch_size = batch_size
        self._lr = init_lr
        self._activation = activation
        self._fc_weight_decay = fc_weight_decay
        self._fc_dropout = fc_dropout
        self._max_epochs = max_epochs
        self._log_rate = log_rate
        self._gpu_options = gpu_options

        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._scaler = None
        self._logger = logging.getLogger(fullname(self))

    def _iterate_minibatches(self, inputs, targets=None, batchsize=64, shuffle=False):
        """
        Iterates over inputs with minibatches
        :param inputs: input dataset, first dimension should be examples
        :param targets: [n_examples, ...] response values, first dimension should be examples
        :param batchsize: batch size
        :param shuffle: flag indicating whether to shuffle the data while making minibatches
        :return: minibatch of (X, Y)
        """
        input_len = inputs.shape[0]
        if shuffle:
            indices = np.arange(input_len)
            np.random.shuffle(indices)
        for start_idx in range(0, input_len, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if targets is None:
                yield inputs[excerpt]
            else:
                yield inputs[excerpt], targets[excerpt]

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

    def _make_behavioral_map(self):
        """
        Makes the temporal mapping function computational graph
        """
        import tensorflow as tf
        from tfutils.model_tool_old import fc as tfutils_fc
        num_classes = len(self._label_mapping.keys())
        assert num_classes == 24
        with self._graph.as_default():
            with tf.variable_scope('behavioral_mapping'):
                out = self._input_placeholder
                out = tf.nn.dropout(out, keep_prob=self._fc_keep_prob, name="dropout_out")
                pred = tfutils_fc(out, 
                                    out_depth=num_classes, 
                                    activation=self._activation, 
                                    batch_norm=False, 
                                    weight_decay=self._fc_weight_decay, name="out")

                self._predictions = pred

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('loss'):

                logits = self._predictions
                
                self.classification_error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._target_placeholder))
                self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                self.total_loss = self.classification_error + self.reg_loss
                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.train.get_or_create_global_step())

    def _init_mapper(self, X, Y):
        """
        Initializes the mapping function graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 1
        with self._graph.as_default():
            self._input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self._target_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
            self._fc_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
            # Build the model graph
            self._make_behavioral_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            self._sess.run(init_op)

    def labels_to_indices(self, labels):
        label2index = OrderedDict()
        indices = []
        for label in labels:
            if label not in label2index:
                label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
            indices.append(label2index[label])
        index2label = OrderedDict((index, label) for label, index in label2index.items())
        return indices, index2label

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        import sklearn
        assert not np.isnan(X).any() and not np.isnan(Y).any()
        self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
        X = self._scaler.transform(X)
        Y, self._label_mapping = self.labels_to_indices(Y.values)
        self.setup()
        assert X.ndim == 2, 'Input matrix rank should be 2.'
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc=' epochs'):
                for counter, batch in enumerate(
                        self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
                    feed_dict = {self._input_placeholder: batch[0],
                                 self._target_placeholder: batch[1],
                                 self._lr_ph: lr,
                                 self._fc_keep_prob: self._fc_dropout}
                    _, loss_value, reg_loss_value = self._sess.run([self.train_op, self.classification_error, self.reg_loss],
                                                                   feed_dict=feed_dict)
                if epoch % self._log_rate == 0:
                    self._logger.debug(f'Epoch: {epoch}, Err Loss: {loss_value:.2f}, Reg Loss: {reg_loss_value:.2f}')

                if loss_value < self._tol:
                    self._logger.debug('Converged.')
                    break

    def predict_proba(self, X):
        assert len(X.shape) == 2, "expected 2-dimensional input"
        assert not np.isnan(X).any()
        X = self._scaler.transform(X)
        with self._graph.as_default():
            preds = []
            for batch in self._iterate_minibatches(X, batchsize=self._batch_size, shuffle=False):
                feed_dict = {self._input_placeholder: batch, self._fc_keep_prob: 1.0}
                preds.append(np.squeeze(self._sess.run([self._predictions], feed_dict=feed_dict)))
            concat_preds = np.concatenate(preds, axis=0)
        proba = tf.nn.softmax(concat_preds)
        # we take only the 0th dimension because the 1st dimension is just the features
        X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                    if array_is_element(dims, X.dims[0])}
        proba = BehavioralAssembly(proba,
                                   coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                   dims=[X.dims[0], 'choice'])
        return proba

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.reset_default_graph()
        self._sess.close()