import sys
minor_version = sys.version_info[1]
if minor_version == 5:
    from keras.layers.convolutional_recurrent import ConvLSTM2D
    from keras.applications.vgg16 import preprocess_input
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, Conv3D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling1D, Input, \
        BatchNormalization, LeakyReLU, Lambda, AveragePooling2D, Reshape
    from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Average, \
        TimeDistributed, Conv2DTranspose, add
    from keras.layers.merge import concatenate
    from keras import backend as K
    from keras.layers import Layer
elif minor_version == 6:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, Conv3D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling1D, Input, \
        BatchNormalization, LeakyReLU, Lambda, AveragePooling2D, Reshape
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, \
        Average, TimeDistributed, Conv2DTranspose, add, Multiply
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Layer


def create_lineage_model(version, input_frames=10, deploy_mode=False, input_channel=None, cell_type='HCT116'):
    '''
    [convlstm1, convlstm2, convlstm2_cnn3x1, convlstm2_relu_cnn3x1, convlstm2_cnn1_2x1, convlstm2_relu_cnn1_2x1, convlstm2_cnn2_2x1, convlstm2_relu_cnn2_2x1, convlstm1_cnn3x1, convlstm1_cnn2_2x1, convlstm1_cnn3_2fc]
    :param version:
    :return:
    '''
    model_note = 'init_state_separate'

    if version == 'convlstm_v7':
        # 'init_state_separate' means the initial state input has the same size with the main input except the frame
        # axis
        model = convlstm_v7(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v70':
        model = convlstm_v70(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v78':
        model = convlstm_v78(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v700':
        model = convlstm_v700(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v701':
        model = convlstm_v701(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v702':
        model = convlstm_v701(input_frames, input_channel=input_channel, pretrain=True)
    elif version == 'convlstm_v703':
        model = convlstm_v703(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v704':
        model = convlstm_v704(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v705':
        model = convlstm_v705(input_frames, input_channel=input_channel)
    elif version == 'convlstm_v783':
        model = convlstm_v783(input_frames, input_channel=input_channel, pretrain=True)
    elif version == 'convlstm_v784':
        model = convlstm_v784(input_frames, input_channel=input_channel, pretrain=True, cell_type=cell_type)
    elif version == 'convlstm_v785':
        model = convlstm_v785(input_frames, input_channel=input_channel, pretrain=True, cell_type=cell_type)
    elif version == 'convlstm_v786':
        model = convlstm_v784(input_frames, input_channel=input_channel, pretrain=True)
        model_note = 'enable_related_seq'
    elif version == 'convlstm_v787':
        model = convlstm_v787(input_frames, input_channel=input_channel, pretrain=True)
    elif version == 'convlstm_v788':
        model = convlstm_v788(input_frames, input_channel=input_channel, pretrain=True)
        model_note = 'enable_related_seq'

    return model, model_note


class Repeat4DTensor(Layer):

    def __init__(self, repeat_times, **kwargs):
        self.repeat_times = repeat_times
        super(Repeat4DTensor, self).__init__(**kwargs)

    def call(self, x):
        x = K.expand_dims(x, 1)
        return K.repeat_elements(x, self.repeat_times, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.repeat_times, input_shape[1], input_shape[2], input_shape[3])




# The v7 family
def convlstm_v7(input_frames, input_channel=3):
    #
    # due to the tensorflow on hpc is version 1.12, which cannot find the ResNet50V2, the model uses the first version
    # of ResNet without identity mapping. Please change this to ResNetV2 when available.

    main_input = Input(shape=(input_frames, 256, 256, input_channel))

    # copy the last hidden state as input and repeat, which form a many-one-many model
    init_state_input = Input(shape=(256, 256, 1))
    init_state = Repeat4DTensor(input_frames)(init_state_input)
    print(init_state.shape)
    r1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                    padding='same', name='reconstruction_1', return_sequences=True)(init_state)
    r1 = BatchNormalization()(r1)


    ev1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_1', return_sequences=True)(main_input)
    ev1 = BatchNormalization()(ev1)
    # ev1 = LeakyReLU(alpha=0.1)(ev1)
    ev2 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_2', go_backwards=True, return_sequences=True)(main_input)
    ev2 = BatchNormalization()(ev2)
    # ev2 = LeakyReLU(alpha=0.1)(ev2)
    # concatenate ev1 and ev2 as the input of ev3
    ev3 = concatenate(inputs=[r1, ev1, ev2], axis=-1)
    ev3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_3', return_sequences=True)(ev3)
    ev3 = BatchNormalization(name='ev3_norm')(ev3)

    # Sequence output
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1', use_bias=False))(ev3)
    x = TimeDistributed(BatchNormalization(name='norm_1'))(x)
    # x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_2'), name='tracking_output_layer')(x)

    model = Model(inputs=[main_input, init_state_input], outputs=output)
    return model


def convlstm_v70(input_frames, input_channel=3):
    # model v70 keeps same with model v7, as control group

    main_input = Input(shape=(input_frames, 256, 256, input_channel))

    # copy the last hidden state as input and repeat, which form a many-one-many model
    init_state_input = Input(shape=(256, 256, 1))
    init_state = Repeat4DTensor(input_frames)(init_state_input)

    # r1 provide the initial location of the target cell
    r1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                    padding='same', name='reconstruction_1', return_sequences=True)(init_state)
    r1 = BatchNormalization()(r1)

    ev1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_1', return_sequences=True)(main_input)
    ev1 = BatchNormalization()(ev1)
    # ev1 = LeakyReLU(alpha=0.1)(ev1)
    ev2 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_2', go_backwards=True, return_sequences=True)(main_input)
    ev2 = BatchNormalization()(ev2)
    # ev2 = LeakyReLU(alpha=0.1)(ev2)
    # concatenate ev1 and ev2 as the input of ev3
    ev3 = concatenate(inputs=[r1, ev1, ev2], axis=-1)
    ev3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_3', return_sequences=True)(ev3)
    ev3 = BatchNormalization(name='ev3_norm')(ev3)

    # Sequence output
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1', use_bias=False), name='time_conv_1')(ev3)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='time_norm_1')(x)
    # x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_2'), name='tracking_output_layer')(x)

    model = Model(inputs=[main_input, init_state_input], outputs=output)
    return model


def convlstm_v78(input_frames, input_channel=3):
    # inherit from model v7, add additional convLSTM

    main_input = Input(shape=(input_frames, 256, 256, input_channel))

    # copy the last hidden state as input and repeat, which form a many-one-many model
    init_state_input = Input(shape=(256, 256, 1))
    init_state = Repeat4DTensor(input_frames)(init_state_input)

    # r1 provide the initial location of the target cell
    r1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                    padding='same', name='reconstruction_1', return_sequences=True)(init_state)
    r1 = BatchNormalization()(r1)


    ev1 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_1', return_sequences=True)(main_input)
    ev1 = BatchNormalization()(ev1)
    # ev1 = LeakyReLU(alpha=0.1)(ev1)
    ev2 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_2', go_backwards=True, return_sequences=True)(main_input)
    ev2 = BatchNormalization()(ev2)
    # ev2 = LeakyReLU(alpha=0.1)(ev2)
    # concatenate ev1 and ev2 as the input of ev3
    ev3 = concatenate(inputs=[r1, ev1, ev2], axis=-1)
    ev3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_3', return_sequences=True)(ev3)
    ev3 = BatchNormalization(name='ev3_norm')(ev3)

    ev4 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                     padding='same', name='ev_4', return_sequences=True)(ev3)
    ev4 = BatchNormalization(name='ev4_norm')(ev4)

    # Sequence output
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1', use_bias=False),
                        name='time_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='time_norm_1')(x)

    output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_last'),
                             name='tracking_output_layer')(x)

    model = Model(inputs=[main_input, init_state_input], outputs=output)
    return model


def convlstm_v700(input_frames, input_channel=3):
    # model v70 keeps same with model v7, as control group

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v70', input_channel=input_channel)
    base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v70-c1-i2'))

    # ev3 = base_model.get_layer('ev3_norm').output

    time_conv_1 = base_model.get_layer('time_conv_1').output
    tracking_output = base_model.get_layer('tracking_output_layer').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='conv_2', use_bias=False))(time_conv_1)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='pooling_2'))(x)
    x = TimeDistributed(BatchNormalization(name='norm_2'))(x)
    x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    x = TimeDistributed(Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='conv_3', use_bias=False))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='pooling_3'))(x)
    x = TimeDistributed(BatchNormalization(name='norm_3'))(x)
    x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    x = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_death'))(x)
    death_output = TimeDistributed(GlobalMaxPooling2D(name='GMP'), name='time_GMP_death')(x)

    # ===Division detection section===
    dx = TimeDistributed(Conv2D(32, (1, 1), strides=(1, 1), padding='same', use_bias=False))(time_conv_1)
    dx = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(dx)
    dx = TimeDistributed(BatchNormalization())(dx)
    dx = TimeDistributed(LeakyReLU(alpha=0.1))(dx)

    dx = TimeDistributed(Conv2D(16, (1, 1), strides=(1, 1), padding='same', use_bias=False))(dx)
    dx = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(dx)
    dx = TimeDistributed(BatchNormalization())(dx)
    dx = TimeDistributed(LeakyReLU(alpha=0.1))(dx)

    dx = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_division'))(dx)
    division_output = TimeDistributed(GlobalMaxPooling2D(name='GMP'), name='time_GMP_division')(dx)

    # ===Reshape section===
    death_output = Lambda(lambda ax: K.squeeze(ax, -1), name='death_output_layer')(death_output)
    division_output = Lambda(lambda ax: K.squeeze(ax, -1), name='division_output_layer')(division_output)
    model = Model(inputs=base_model.inputs, outputs=[tracking_output, death_output, division_output])
    return model


def convlstm_v701(input_frames, input_channel=3, pretrain=False):
    # model v70's family, predict death only

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v70', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v70-c1-i2'))

    # ev3 = base_model.get_layer('ev3_norm').output

    time_conv_1 = base_model.get_layer('time_conv_1').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='conv_2', use_bias=False))(time_conv_1)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='pooling_2'))(x)
    x = TimeDistributed(BatchNormalization(name='norm_2'))(x)
    x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    x = TimeDistributed(Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='conv_3', use_bias=False))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='pooling_3'))(x)
    x = TimeDistributed(BatchNormalization(name='norm_3'))(x)
    x = TimeDistributed(LeakyReLU(alpha=0.1))(x)

    x = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_death'))(x)
    death_output = TimeDistributed(GlobalMaxPooling2D(name='GMP'), name='time_GMP_death')(x)

    # ===Reshape section===
    death_output = Lambda(lambda ax: K.squeeze(ax, -1), name='death_output_layer')(death_output)
    model = Model(inputs=base_model.inputs, outputs=death_output)
    return model

def convlstm_v703(input_frames, input_channel=3, pretrain=False):
    # model v70's family, predict division only

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v70', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v70-c1-i2'))

    # ev3 = base_model.get_layer('ev3_norm').output

    time_conv_1 = base_model.get_layer('time_conv_1').output

    # ===Division detection section===
    dx = TimeDistributed(Conv2D(32, (1, 1), strides=(1, 1), padding='same', use_bias=False))(time_conv_1)
    dx = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(dx)
    dx = TimeDistributed(BatchNormalization())(dx)
    dx = TimeDistributed(LeakyReLU(alpha=0.1))(dx)

    dx = TimeDistributed(Conv2D(16, (1, 1), strides=(1, 1), padding='same', use_bias=False))(dx)
    dx = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(dx)
    dx = TimeDistributed(BatchNormalization())(dx)
    dx = TimeDistributed(LeakyReLU(alpha=0.1))(dx)

    dx = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_division'))(
        dx)
    division_output = TimeDistributed(GlobalMaxPooling2D(name='GMP'), name='time_GMP_division')(dx)

    # ===Reshape section===
    division_output = Lambda(lambda ax: K.squeeze(ax, -1), name='division_output_layer')(division_output)
    model = Model(inputs=base_model.inputs, outputs=division_output)
    return model


def convlstm_v704(input_frames, input_channel=3, pretrain=False):
    # model v70's family, only predict division spatially

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v70', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v70-c1-i2'))

    time_norm_1 = base_model.get_layer('time_norm_1').output

    # ===Division detection section===
    division_output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same'), name='division_output_layer')(
        time_norm_1)

    model = Model(inputs=base_model.inputs, outputs=division_output)
    return model


def convlstm_v705(input_frames, input_channel=3, pretrain=False):
    # model v70's family, only predict death spatially

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v70', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v70-c1-i2'))

    time_norm_1 = base_model.get_layer('time_norm_1').output

    # ===Division detection section===
    death_output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same'), name='death_output_layer')(time_norm_1)

    model = Model(inputs=base_model.inputs, outputs=death_output)
    return model


def convlstm_v783(input_frames, input_channel=3, pretrain=False):
    # model v78's family, only predict death spatially

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v78', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i2-gss-con-1_check'))

    ev4 = base_model.get_layer('ev4_norm').output

    tracking_output = base_model.get_layer('tracking_output_layer').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1'), name='death_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='death_norm_1')(x)

    death_output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_last'),
                             name='death_output_layer')(x)

    model = Model(inputs=base_model.inputs, outputs=[tracking_output, death_output])
    return model


def convlstm_v784(input_frames, input_channel=3, pretrain=False, cell_type='HCT116'):
    # model v78's family, only predict death spatially

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v78', input_channel=input_channel)
    if pretrain:
        if cell_type == 'HCT116':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i2-gss-con-1_check'))
        elif cell_type == 'MCF7':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i10-gss-MCF7-con-1_check'))
        elif cell_type == 'HeLa':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c1-gss-HeLa-1'))
        elif cell_type == 'U2OS':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-gss-U2OS-1_check'))

    ev4 = base_model.get_layer('ev4_norm').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1'), name='division_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='division_norm_1')(x)

    division_output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same'),
                                      name='division_output_layer')(x)

    model = Model(inputs=base_model.inputs, outputs=division_output)
    return model


def convlstm_v785(input_frames, input_channel=3, pretrain=False, cell_type='HCT116'):
    # model v78's family, only predict death spatially

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v78', input_channel=input_channel)
    if pretrain:
        if cell_type == 'HCT116':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i2-gss-con-1_check'))
        elif cell_type == 'MCF7':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i10-gss-MCF7-con-1_check'))
        elif cell_type == 'U2OS':
            base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-gss-U2OS-1_check'))

    ev4 = base_model.get_layer('ev4_norm').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1'), name='death_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='death_norm_1')(x)

    death_output = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid', strides=(1, 1), padding='same', name='conv_last'),
                             name='death_output_layer')(x)

    model = Model(inputs=base_model.inputs, outputs=death_output)
    return model


def convlstm_v787(input_frames, input_channel=3, pretrain=False):
    # model v78's family, only predict death spatially with multiple output channels

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v78', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i2-gss-con-1_check'))

    ev4 = base_model.get_layer('ev4_norm').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1'), name='death_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='death_norm_1')(x)

    death_output = TimeDistributed(Conv2D(2, (1, 1), activation='sigmoid', strides=(1, 1), padding='same'),
                                      name='death_output_layer')(x)

    model = Model(inputs=base_model.inputs, outputs=death_output)
    return model


def convlstm_v788(input_frames, input_channel=3, pretrain=False):
    # model v78's family, only predict division spatially with multiple output channels

    # loading layers in v70 before ev3
    base_model, _ = create_lineage_model('convlstm_v78', input_channel=input_channel)
    if pretrain:
        base_model.load_weights('{}{}.h5'.format('./benchmarks/', 'convlstm_v78-c3-i2-gss-con-1_check'))

    ev4 = base_model.get_layer('ev4_norm').output

    # ===Death detection section===
    x = TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv_1'), name='division_conv_1')(ev4)
    x = TimeDistributed(BatchNormalization(name='norm_1'), name='division_norm_1')(x)

    division_output = TimeDistributed(Conv2D(3, (1, 1), activation='sigmoid', strides=(1, 1), padding='same'),
                                      name='division_output_layer')(x)

    model = Model(inputs=base_model.inputs, outputs=division_output)
    return model