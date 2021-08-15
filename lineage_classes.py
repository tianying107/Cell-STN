import numpy as np
from scipy import signal
import utility as ut
import sys
minor_version = sys.version_info[1]
import tensorflow as tf
if minor_version == 5:
    import keras
    from keras import backend as K
elif minor_version ==6:
    import tensorflow.keras as keras
    from tensorflow.keras import backend as K


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, sub_seqs, annotations, output_shape, batch_size=4, input_dim=(10,256,256,3),
                 shuffle=True, rotate=True, speedup=False, convlstm_loc='conv_last', init_paras={}, gt_paras={},
                 use_sample=None):
        """

        :param sub_seqs:
        :param annotations:
        :param output_shape:
        :param batch_size:
        :param input_dim:
        :param shuffle: (Bool) Enable data augmentation: shuffle
        :param rotate: (Bool) Enable data augmentation: rotate
        :param speedup: (Bool) Enable data augmentation: speedup
        :param convlstm_loc:
        :param init_paras:
        :param gt_paras:
        :param use_sample: (int) This variable is used for Global Samples Shuffle (GSS). When GSS is enabled, the
        generator will be initialized with the entire data set. However, the generator will only load partial of the
        data in use. For example, when this generator is used for training, the training data for each epoch will be
        different. In order to use this feature, the variable shuffle should be True.
        """

        print('Data Generator Initialization')
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.batch_size = batch_size

        self.sub_seqs = sub_seqs
        self.annotations = annotations

        self.shuffle = shuffle
        self.rotate = rotate
        self.speedup = speedup

        self.init_mask = ((convlstm_loc == 'init_mask') | (convlstm_loc == 'init_state_separate'))
        self.init_state_separate = (convlstm_loc == 'init_state_separate')
        self.convlstm_loc = convlstm_loc

        self.init_repeat = (convlstm_loc == 'init_repeat')

        self.init_paras = init_paras
        self.gt_paras = gt_paras

        if use_sample is None:
            self.samples_num = len(self.sub_seqs)
        else:
            self.samples_num = use_sample

        self.configuration = ut.read_configuration()
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples_num / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        sub_seqs_batch = [self.sub_seqs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(sub_seqs_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sub_seqs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.indexes = self.indexes[0:self.samples_num]

    def __data_generation(self, sub_seqs_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *input_dim)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        if self.init_repeat:
            # in this model, the init_state is assigned a map for the first frame, and padding zeros for the rest frames
            init_state = np.empty((self.batch_size, *self.input_dim[:]), dtype=float)
        elif self.init_mask:
            # in init_mask mode, the init_state only has one channel, which indicate the initial location of the
            # target cell
            init_state = np.empty((self.batch_size, *self.input_dim[1:-1], 1), dtype=float)
        elif self.convlstm_loc == 'conv_first':
            init_state = np.empty((self.batch_size, *self.input_dim[1:-1], 16), dtype=float)
        else:
            init_state = np.empty((self.batch_size, *self.output_shape[2::]), dtype=float)
        y = np.empty((self.batch_size, *self.output_shape[1::]), dtype=float)

        # Generate data
        for i, sub_seq in enumerate(sub_seqs_batch):
            # load data for the entire experiment
            temp_images = ut.read_image_array(self.configuration, sub_seq)
            ground_truth = ut.select_annotation_for_subseq(self.annotations, sub_seq)

            init_state[i,] = ut.generate_init_state(ground_truth, sub_seq, self.input_dim, self.output_shape,
                                                    self.convlstm_loc, init_mask=self.init_mask)
            # Data augmentations
            # 1. speedup
            speed_factor = int(np.random.randint(1, 3, 1))  # allowed speed factor 1 or 2
            speedup = self.speedup & ut.input_sequence_speedup_check(sub_seq, self.input_dim, temp_images, ground_truth,
                                                                     speed_factor=speed_factor)
            if speedup:
                data = ut.generate_input_sequence(sub_seq, self.input_dim, temp_images, speed_factor=speed_factor)
                y[i,] = ut.generate_ground_truth_sequence(ground_truth, sub_seq, self.input_dim, self.output_shape,
                                                          speed_factor=speed_factor, kwargs=self.gt_paras)
            else:
                # generate input sequence without speedup
                data = ut.generate_input_sequence(sub_seq, self.input_dim, temp_images)
                # generate ground truth without speedup
                y[i,] = ut.generate_ground_truth_sequence(ground_truth, sub_seq, self.input_dim, self.output_shape,
                                                          kwargs=self.gt_paras)

            # 2. rotate input, ground truth, init state
            rotate = int(np.random.randint(0, 4, 1))
            if self.rotate:
                data = np.rot90(data, rotate, (1, 2))
                y[i, ] = np.rot90(y[i, ], rotate, (1, 2))
                init_state[i, ] = np.rot90(init_state[i, ], rotate, (0, 1))
            X[i, ] = data

            # middle step visualization for debug
            debug = False
            if debug and not ut.deploy_check():
                if speedup:
                    print('Augmentation with speedup: {}'.format(speedup))
                    print('Speed factor: {}'.format(speed_factor))
                # import pylab as plt
                # fig, axs = plt.subplots(2, 5)
                # for f in range(y[i, ].shape[0]):
                #     axs[f // 5][f % 5].imshow(X[i, f, :, :, 1], cmap='gray')
                #     axs[f // 5][f % 5].imshow(y[i, f, :, :, 0], alpha=0.2)
                # plt.show()
                exit()

        # return X, y
        if self.init_state_separate:
            return [X, init_state], y
        elif self.init_mask:
            X[:, 0, ] = np.multiply(X[:, 0, ], init_state)
            return X, y
        else:
            return [X, init_state], y


class MultitaskDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, sub_seqs, annotations, output_shapes, batch_size=4, input_dim=(10,256,256,3),
                 shuffle=True, rotate=True, speedup=False, model_note='', init_paras={}, gt_paras={},
                 use_sample=None):
        '''
        Initialization
        '''

        print('Initialization')
        self.input_dim = input_dim
        self.output_shapes = output_shapes
        self.batch_size = batch_size

        self.sub_seqs = sub_seqs
        self.annotations = annotations

        self.shuffle = shuffle
        self.rotate = rotate
        self.speedup = speedup

        self.model_note = model_note

        self.init_paras = init_paras
        self.gt_paras = gt_paras

        if use_sample is None:
            self.samples_num = len(self.sub_seqs)
        else:
            self.samples_num = use_sample

        self.configuration = ut.read_configuration()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples_num / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        sub_seqs_batch = [self.sub_seqs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(sub_seqs_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sub_seqs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.indexes = self.indexes[0:self.samples_num]

    def __data_generation(self, sub_seqs_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *input_dim)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        init_state = np.empty((self.batch_size, *self.input_dim[1:-1], 1), dtype=float)

        if 'tracking' in self.output_shapes:
            tracking_output = np.empty((self.batch_size, *self.output_shapes['tracking'][1::]), dtype=float)
        if 'death' in self.output_shapes:
            death_output = np.empty((self.batch_size, *self.output_shapes['death'][1::]), dtype=float)
        if 'division' in self.output_shapes:
            division_output = np.empty((self.batch_size, *self.output_shapes['division'][1::]), dtype=float)

        # Generate data
        for i, sub_seq in enumerate(sub_seqs_batch):
            # in_scope check for death
            if 'death' in self.output_shapes:
                ut.death_sequence_valid_check(sub_seq, self.input_dim, self.annotations)

            # load data for the entire experiment
            temp_images = ut.read_image_array(self.configuration, sub_seq)
            data = ut.generate_input_sequence(sub_seq, self.input_dim, temp_images)

            # rotate input
            rotate = int(np.random.randint(0, 4, 1))
            if self.rotate:
                data = np.rot90(data, rotate, (1, 2))

            X[i, ] = data

            ground_truth = ut.select_annotation_for_subseq(self.annotations, sub_seq)
            # tracking ground truth session
            if 'tracking' in self.output_shapes:
                tracking_output[i, ] = ut.generate_ground_truth_sequence(ground_truth, sub_seq, self.input_dim,
                                                                         self.output_shapes['tracking'],
                                                                         kwargs=self.gt_paras)
                if self.rotate:
                    tracking_output[i, ] = np.rot90(tracking_output[i,], rotate, (1, 2))
            # death ground truth session
            if 'death' in self.output_shapes:
                death_output[i, ] = ut.generate_death_truth_sequence(ground_truth, sub_seq, self.input_dim,
                                                                     self.output_shapes['death'],
                                                                     spatial=(len(self.output_shapes['death']) > 2))
                if self.rotate and (len(self.output_shapes['death']) > 2):
                    death_output[i, ] = np.rot90(death_output[i,], rotate, (1, 2))
            # division ground truth session
            if 'division' in self.output_shapes:
                kwargs = {}
                if self.model_note == 'enable_related_seq':
                    kwargs = {'annotations': self.annotations}
                division_output[i, ] = ut.generate_division_truth_sequence(ground_truth, sub_seq, self.input_dim,
                                                                           self.output_shapes['division'], spatial=
                                                                           (len(self.output_shapes['division']) > 2),
                                                                           kwargs=kwargs
                                                                           )
                if self.rotate and len(self.output_shapes['division']) > 2:
                    division_output[i, ] = np.rot90(division_output[i,], rotate, (1, 2))

            # initial state session
            init_state[i,] = ut.generate_init_state(ground_truth, sub_seq, self.input_dim, self.output_shapes,
                                                    self.model_note, init_mask=True)
            if self.rotate:
                init_state[i,] = np.rot90(init_state[i, ], rotate, (0, 1))

            # debug = True
            # if debug and not ut.deploy_check():
            #     import pylab as plt
            #     fig, axs = plt.subplots(2, 5)
            #     print(sub_seq)
            #     # for f in range(division_output[i,].shape[0]):
            #     #     axs[f // 5][f % 5].imshow(X[i, f, :, :, 1], cmap='gray')
            #     #     axs[f // 5][f % 5].imshow(division_output[i, f, :, :, 0], cmap='jet', alpha=0.4)
            #     death_y = np.zeros([*death_output.shape[0:-1], death_output.shape[-1] + 1])
            #     death_y[:, :, :, :, 0] = death_output[:, :, :, :, 0]
            #     death_y[:, :, :, :, 1] = death_output[:, :, :, :, 1]
            #     for f in range(death_output[i,].shape[0]):
            #         axs[f // 5][f % 5].imshow(X[i, f, :, :, 1], cmap='gray')
            #         axs[f // 5][f % 5].imshow(death_y[i, f, :, :, :], cmap='jet', alpha=0.1)
            #     plt.show()
            #     exit()

        outputs = []
        if 'tracking' in self.output_shapes:
            outputs.append(tracking_output)
        if 'death' in self.output_shapes:
            outputs.append(death_output)
        if 'division' in self.output_shapes:
            outputs.append(division_output)
        if len(outputs) == 1:
            outputs = outputs[0]

        return [X, init_state], outputs


def weighted_binary_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = [0.5,2] # Class one at 0.5, class 2 twice the normal weights.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(np.array(weights))

    def loss(y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)
        # print(b_ce.shape)
        # Apply the weights
        weight_vector = y_true * weights[1] + (1. - y_true) * weights[0]
        weighted_b_ce = weight_vector * b_ce
        # print(weighted_b_ce)

        # Return the mean error
        return K.mean(weighted_b_ce)

    return loss


def sorted_binary_crossentropy():
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = [0.5,2] # Class one at 0.5, class 2 twice the normal weights.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    def loss(y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce_1 = K.mean(K.binary_crossentropy(y_true[:,:,:,:,0], y_pred[:,:,:,:,0]))

        b_ce_2_1 = K.mean(K.binary_crossentropy(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1]), keepdims=True)
        b_ce_2_1 = K.reshape(b_ce_2_1, shape=(1,))

        b_ce_2_2 = K.mean(K.binary_crossentropy(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 2]), keepdims=True)
        b_ce_2_2 = K.reshape(b_ce_2_2, shape=(1,))

        b_ce_2_3 = K.mean(K.binary_crossentropy(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 1]), keepdims=True)
        b_ce_2_3 = K.reshape(b_ce_2_3, shape=(1,))

        b_ce_2_4 = K.mean(K.binary_crossentropy(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2]), keepdims=True)
        b_ce_2_4 = K.reshape(b_ce_2_4, shape=(1,))

        # # previous code v1
        # b_ce_2_1_m = K.min(K.concatenate((b_ce_2_1, b_ce_2_3), axis=-1))
        # b_ce_2_2_m = K.min(K.concatenate((b_ce_2_2, b_ce_2_4), axis=-1))
        #
        # y_true_child = K.maximum(y_true[:, :, :, :, 1], y_true[:, :, :, :, 2])
        # y_pred_child = K.maximum(y_pred[:, :, :, :, 1], y_pred[:, :, :, :, 2])
        # b_ce_child = K.mean(K.binary_crossentropy(y_true_child, y_pred_child))
        # return b_ce_1 + b_ce_2_1_m + b_ce_2_2_m + b_ce_child

        # # previous code v2
        # b_ce_2_1_m = K.min(K.concatenate((b_ce_2_1, b_ce_2_2), axis=-1))
        # b_ce_2_2_m = K.min(K.concatenate((b_ce_2_3, b_ce_2_4), axis=-1))
        #
        # y_true_diff = K.abs(tf.subtract(y_true[:, :, :, :, 1], y_true[:, :, :, :, 2]))
        # y_pred_diff = K.abs(tf.subtract(y_pred[:, :, :, :, 1], y_pred[:, :, :, :, 2]))
        # b_ce_diff = K.mean(K.binary_crossentropy(y_true_diff, y_pred_diff))
        # return b_ce_1 + b_ce_2_1_m + b_ce_2_2_m + b_ce_diff

        # new code
        b_ce_2_1_m = K.min(K.concatenate((b_ce_2_1, b_ce_2_2), axis=-1))
        b_ce_2_2_m = K.min(K.concatenate((b_ce_2_3, b_ce_2_4), axis=-1))
        b_ce_2_3_m = K.min(K.concatenate((b_ce_2_1, b_ce_2_3), axis=-1))
        b_ce_2_4_m = K.min(K.concatenate((b_ce_2_2, b_ce_2_4), axis=-1))

        y_true_diff = K.abs(tf.subtract(y_true[:, :, :, :, 1], y_true[:, :, :, :, 2]))
        y_pred_diff = K.abs(tf.subtract(y_pred[:, :, :, :, 1], y_pred[:, :, :, :, 2]))
        b_ce_diff = K.mean(K.binary_crossentropy(y_true_diff, y_pred_diff))


        # print(K.mean(K.mean(K.mean(b_ce, 1), 1), 1).shape)
        # Return the mean error
        return b_ce_1 + b_ce_2_1_m + b_ce_2_2_m + b_ce_2_3_m + b_ce_2_4_m + b_ce_diff

    return loss


def weighted_peak_crossentropy(weights=[1, 1]):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = [0.5,2] # Class one at 0.5, class 2 twice the normal weights.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(np.array(weights))

    def loss(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.mean(K.binary_crossentropy(y_true, y_pred))
        # print(b_ce.shape)
        peak_mse = adjusted_RMSE(y_true, y_pred)

        # Return the mean error
        return weights[1] * b_ce + weights[0] * peak_mse

    return loss


def weighted_accuracy_crossentropy(weights=[1, 1]):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = [0.5,2] # Class one at 0.5, class 2 twice the normal weights.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(np.array(weights))

    def loss(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.mean(K.binary_crossentropy(y_true, y_pred))
        # print(b_ce.shape)
        peak_accuracy = adjusted_peak_accuracy(y_true, y_pred)

        # Return the mean error
        return weights[1] * b_ce + weights[0] * (1 - peak_accuracy)

    return loss


def sequence_binary_crossentropy(weight_steps=1):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = [0.5,2] # Class one at 0.5, class 2 twice the normal weights.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """



    def loss(y_true, y_pred):
        b_ce = K.binary_crossentropy(y_true, y_pred)
        weight = 1 + weight_steps * np.arange(b_ce.shape[1])
        weighted_b_ce = tf.multiply(b_ce, weight[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis])
        # Return the mean error
        return K.mean(weighted_b_ce)

    return loss


def RMSE(y_true, y_pred):

    pred_cols = K.argmax(K.max(y_pred, axis=2), axis=2)
    pred_rows = K.argmax(K.max(y_pred, axis=3), axis=2)

    true_cols = K.argmax(K.max(y_true, axis=2), axis=2)
    true_rows = K.argmax(K.max(y_true, axis=3), axis=2)

    row_sq = K.square(pred_rows - true_rows)
    col_sq = K.square(pred_cols - true_cols)

    # sq_sum = row_sq + col_sq
    sq_sum = K.squeeze(row_sq + col_sq, axis=-1)

    sq_sum = K.cast(sq_sum, dtype='float64')
    sqrt_sum = K.sqrt(sq_sum)

    RMSE = K.mean(sqrt_sum)

    return RMSE


def adjusted_RMSE(y_true, y_pred):

    pred_cols = K.argmax(K.max(y_pred, axis=2), axis=2)
    pred_rows = K.argmax(K.max(y_pred, axis=3), axis=2)

    true_cols = K.argmax(K.max(y_true, axis=2), axis=2)
    true_rows = K.argmax(K.max(y_true, axis=3), axis=2)

    row_sq = K.square(pred_rows - true_rows)
    col_sq = K.square(pred_cols - true_cols)

    # sq_sum = row_sq + col_sq
    sq_sum = K.squeeze(row_sq + col_sq, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)

    in_scope = K.cast(K.equal(K.max(K.max(y_true, axis=2), axis=2), 1), dtype='float32')
    sq_sum = K.cast(sq_sum, dtype='float32')
    sq_sum = K.sqrt(sq_sum)
    sq_sum = tf.multiply(sq_sum, in_scope)
    # sq_sum = K.cast(sq_sum, dtype='float64')
    # sqrt_sum = K.sqrt(sq_sum)

    RMSE = K.mean(sq_sum)

    return RMSE


def peak_accuracy(y_true, y_pred):
    true_positive_threshold = 10
    pred_cols = K.argmax(K.max(y_pred, axis=2), axis=2)
    pred_rows = K.argmax(K.max(y_pred, axis=3), axis=2)

    true_cols = K.argmax(K.max(y_true, axis=2), axis=2)
    true_rows = K.argmax(K.max(y_true, axis=3), axis=2)

    row_sq = K.square(pred_rows - true_rows)
    col_sq = K.square(pred_cols - true_cols)

    sq_sum = K.squeeze(row_sq + col_sq, axis=-1)
    sq_sum = K.cast(sq_sum, dtype='float64')
    sq_sum = K.sqrt(sq_sum)

    true_positives = K.cast(K.less_equal(sq_sum, true_positive_threshold), dtype='float64')

    return K.mean(true_positives)


def adjusted_peak_accuracy(y_true, y_pred):
    true_positive_threshold = 10
    pred_cols = K.argmax(K.max(y_pred, axis=2), axis=2)
    pred_rows = K.argmax(K.max(y_pred, axis=3), axis=2)

    true_cols = K.argmax(K.max(y_true, axis=2), axis=2)
    true_rows = K.argmax(K.max(y_true, axis=3), axis=2)

    row_sq = K.square(pred_rows - true_rows)
    col_sq = K.square(pred_cols - true_cols)

    sq_sum = K.squeeze(row_sq + col_sq, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)

    in_scope = K.cast(K.equal(K.max(K.max(y_true, axis=2), axis=2), 1), dtype='float32')
    sq_sum = K.cast(sq_sum, dtype='float32')
    sq_sum = K.sqrt(sq_sum)
    sq_sum = tf.multiply(sq_sum, in_scope)
    true_positives = K.cast(K.less_equal(sq_sum, true_positive_threshold), dtype='float32')

    return K.mean(true_positives)


def event_peak_accuracy(y_true, y_pred):
    if len(y_true.get_shape().as_list()) > 2:
        true_positive_threshold = 1
        pred_frame = K.argmax(K.max(K.max(y_pred, axis=2), axis=2), axis=1)
        gt_frame = K.argmax(K.max(K.max(y_true, axis=2), axis=2), axis=1)
        distance = K.abs(pred_frame - gt_frame)
        distance = distance[:, 0]

        in_scope = K.cast(K.equal(K.max(K.max(K.max(y_true, axis=2), axis=2), axis=1), 1), dtype='float32')
        in_scope = in_scope[:, 0]
        distance = K.cast(distance, dtype='float32')
        distance = tf.multiply(distance, in_scope)

        true_positives = K.cast(K.less_equal(distance, true_positive_threshold), dtype='float32')
        
        return K.mean(true_positives)
    else:
        # binary accuracy
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def event_sequence_accuracy(y_true, y_pred):
    if len(y_true.get_shape().as_list()) > 2:
        value_true_positive_threshold = 0.5
        pred_value = K.max(K.max(K.max(y_pred, axis=2), axis=2), axis=1)
        pred_value = K.cast(K.greater(pred_value, value_true_positive_threshold), dtype='float32')
        gt_value = K.max(K.max(K.max(y_true, axis=2), axis=2), axis=1)
        gt_value = K.cast(K.greater(gt_value, value_true_positive_threshold), dtype='float32')

        positives = K.equal(gt_value, pred_value)
        positives = positives[:, 0]

        return K.mean(positives, axis=-1)
    else:
        # binary accuracy
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
