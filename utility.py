from utilities import image2array, coordinate2map, ground_truth_convert, settings, evaluate_utils
import numpy as np
import os
import pickle
import json
import random


def alloc_settings_command(commands):
    """

    :param commands: Input commands from sys.argv to define the settings
    :return: (dict) The dictionary contains variables mentioned in commands
    """
    return settings.alloc_settings_command(commands)


def get_settings(command_dict):
    """

    :param command_dict: (dict) The dictionary that indicates variables need to by modified beyond default. Can be
    generated by alloc_settings_command.
    :return: multiple variables
    """
    return settings.get_settings(**command_dict)


def read_configuration(file_path=None):
    """

    :param file_path: The path of the configuration file.
    :return:
    """
    if file_path is None:
        file_path = './Config.json'
    with open(file_path) as f:
        configuration = json.load(f)
    return configuration


def generate_temp_data(cell_type, configuration=None):
    """
    Convert the raw tiff image to np array and stored in temp directory
    :param cell_type: (string) the used cell type
    :param configuration: (dictionary) config information
    :return: None
    """
    if configuration is None:
        configuration = read_configuration()
    image2array.tif2npy(cell_type, configuration)


def deploy_check(configuration=None):
    """
    check if the current environment is in the deploy mode
    :param configuration: (dictionary) config information
    :return: (bool) the current environment is in the deploy mode
    """
    if configuration is None:
        configuration = read_configuration()
    return configuration['production'] == 'deploy'


def temp_dataset_check(cell_type, configuration=None):
    """
    check if the npy format dataset exist in the current environment
    :param cell_type: (string) the used cell type
    :param configuration: (dictionary) config information
    :return: (bool) the npy format dataset exist in the current environment
    """
    if configuration is None:
        configuration = read_configuration()
    temp_image_path = '{}{}/'.format(configuration['temp_image_path'], cell_type)
    if cell_type not in configuration['cell_types']:
        print('The cell type is not defined in the configuration file. Please check the config file and try again.')
        exit()
    if os.path.isdir(temp_image_path):
        for experiment in configuration[cell_type]['experiments']:
            if not os.path.isfile('{}Pos{}_c{}f{}.npy'.format(temp_image_path, experiment['id'], experiment['channels'],
                                                              experiment['frames'])):
                return False
    return os.path.isdir(temp_image_path)


def generate_ground_truth_sequence(ground_truth, sub_seq, input_dim, output_shape, speed_factor=1, kwargs={}):
    """
    This function generates the ground truth confidence maps for training. The type of the generated map is square map
    by default. The input should have specified one target cell.
    :param ground_truth: (dictionary) {'rows': list, 'cols': list} records the position of the target cell at all time
    points
    :param sub_seq: (dictionary) {'offset_x': int, 'offset_y': int, 'start_frame': int} indicates the starting point of
    the target cell
    :param input_dim: (list) (frames, rows, cols, channels) indicates the input shape of the model.
    :param output_shape: (list) (batch, frames, rows, cols, channels) indicates the output shape of the model
    :return: sequence: (np_array) [frames, rows, cols, channels] The first channel represents the position of the cell
    in normal state. The second channel expresses the last known position before the cell went out of scope.
    """
    
    start = sub_seq['start_frame']
    frames = input_dim[0]
    rows = ground_truth['rows'][start:start+speed_factor*frames:speed_factor]
    cols = ground_truth['cols'][start:start+speed_factor*frames:speed_factor]

    mask_ground_truth = False
    if 'mask_ground_truth' in kwargs:
        mask_ground_truth = kwargs.pop('mask_ground_truth')

    sequence = np.zeros(output_shape[1::])

    for f, coordinate in enumerate(zip(rows, cols)):
        coordinate_correct = ground_truth_convert.ground_truth_scaling(coordinate[0], coordinate[1],
                                                                       sub_seq['offset_x'], sub_seq['offset_y'],
                                                                       input_dim, output_shape)
        if output_shape[-1] == 1:
            if mask_ground_truth:
                sequence[f, :, :, 0] = ground_truth_convert.select_mask(sub_seq, coordinate, f)
            else:
                sequence[f, :, :, 0] = ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, output_shape,
                                                                                      type='square', **kwargs)
        elif output_shape[-1] > 1:
            if ground_truth_convert.out_of_scope_check(coordinate_correct, output_shape):
                # if out of scope put the previous confidence map into the second channel
                sequence[f, :, :, 1] = sequence[f-1, :, :, 0]
            else:
                sequence[f, :, :, 0] = ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, output_shape,
                                                                                          type='square')

    return sequence


def generate_death_truth_sequence(ground_truth, sub_seq, input_dim, output_shape, speed_factor=1, spatial=False, kwargs={}):
    """
    This function generates the ground truth death sequence for training. This sequence contains one binary indicator
    for each frame to demonstrate if the target cell dead at this specific frame. The input should have specified one
    target cell.
    :param ground_truth: (dictionary) {'rows': list, 'cols': list} records the position of the target cell at all time
    points
    :param sub_seq: (dictionary) {'offset_x': int, 'offset_y': int, 'start_frame': int} indicates the starting point of
    the target cell
    :param output_shape: (list) (batch, frames) indicates the output shape of the model
    :param speed_factor: (init) factor of speedup the input sequence with default value of 1. When the factor is 1, the
    input is at normal speed. A factor of 2 means the frequency of the input is 2 times than the normal one.
    :param spatial: (bool) generate spatial map sequence or binary sequence as ground truth.
    :param kwargs: Additional information for future development
    :return: sequence: (np_array) [frames]
    """

    start = sub_seq['start_frame']
    frames = output_shape[1]
    output_channels = output_shape[-1]
    sequence = ground_truth['death'][start:start+speed_factor*frames:speed_factor]
    if spatial:
        spatial_sequence = generate_ground_truth_sequence(ground_truth, sub_seq, input_dim, output_shape,
                                                          kwargs=kwargs)
        sequence = np.reshape(sequence, (*sequence.shape, 1, 1, 1))
        sequence = np.multiply(spatial_sequence, sequence)
        if output_channels == 2:
            sequence[:, :, :, 1] = spatial_sequence[:, :, :, 0]
    return sequence


def generate_division_truth_sequence(ground_truth, sub_seq, input_dim, output_shape, speed_factor=1, spatial=False, kwargs={}):
    """
    This function generates the ground truth division sequence for training. This sequence contains one binary indicator
    for each frame to demonstrate if the target cell dead at this specific frame. The input should have specified one
    target cell.
    :param ground_truth: (dictionary) {'rows': list, 'cols': list} records the position of the target cell at all time
    points
    :param sub_seq: (dictionary) {'offset_x': int, 'offset_y': int, 'start_frame': int} indicates the starting point of
    the target cell
    :param input_dim: (list) (frames, rows, cols, channels) indicates the input shape of the model.
    :param output_shape: (list) (batch, frames) indicates the output shape of the model
    :param speed_factor: (init) factor of speedup the input sequence with default value of 1. When the factor is 1, the
    input is at normal speed. A factor of 2 means the frequency of the input is 2 times than the normal one.
    :param spatial: (bool) generate spatial map sequence or binary sequence as ground truth.
    :param kwargs: For other keyword-only arguments. If the annotations is given in this variable, the returned ground
    truth will include both given sub seq and its related sub seq.
    :return: sequence: (np_array) [frames]
    """

    start = sub_seq['start_frame']
    frames = output_shape[1]
    output_channels = output_shape[-1]
    sequence = np.array(ground_truth['division'][start:start+speed_factor*frames:speed_factor])
    if spatial:
        # when the annotations is none, there is no need to find the related ground truth
        annotations = kwargs.pop('annotations', None)

        spatial_sequence = generate_ground_truth_sequence(ground_truth, sub_seq, input_dim, output_shape,
                                                          kwargs=kwargs)

        related_spatial_sequence = np.zeros(spatial_sequence.shape)
        related_ground_truth, related_sub_seq = ground_truth_convert.get_related_sub_seq(annotations, sub_seq, output_shape)
        if related_sub_seq is not None:
            if output_channels == 1:
                # find out the next frame
                peak_frame = np.squeeze(np.where(sequence == 1))
                if peak_frame + 1 < len(sequence):
                    sequence[peak_frame + 1] = 1
                related_spatial_sequence = generate_ground_truth_sequence(related_ground_truth, related_sub_seq, input_dim,
                                                                          output_shape, kwargs=kwargs)
            elif output_channels == 3:
                # find out the next frame
                peak_frame = np.squeeze(np.where(sequence == 1))
                related_sequence = np.zeros(sequence.shape, dtype=sequence.dtype)
                if peak_frame + 1 < len(sequence):
                    sequence[peak_frame + 1] = 1
                    related_sequence[peak_frame + 1] = 1
                related_spatial_sequence[:, :, :, 1] = generate_ground_truth_sequence(related_ground_truth,
                                                                                      related_sub_seq, input_dim,
                                                                                      output_shape,
                                                                                      kwargs=kwargs)[:, :, :, 0]
                related_spatial_sequence[:, :, :, 2] = spatial_sequence[:, :, :, 0]
                related_sequence = np.reshape(related_sequence, (*related_sequence.shape, 1, 1, 1))
                related_spatial_sequence = np.multiply(related_spatial_sequence, related_sequence)
                spatial_sequence = np.multiply(spatial_sequence, 1-related_sequence)

        sequence = np.reshape(sequence, (*sequence.shape, 1, 1, 1))
        sequence = np.multiply(np.maximum(spatial_sequence, related_spatial_sequence), sequence)

    return sequence


def input_sequence_speedup_check(sub_seq, input_dim, temp_images, ground_truth, speed_factor=1):
    """
    This function check if the given sub_seq is valid for speedup process, in terms of length, contain division and contain death
    :param sub_seq:
    :param input_dim:
    :param temp_images:
    :param ground_truth:
    :param speed_factor:
    :return: (bool)
    """
    start_f = sub_seq['start_frame']
    frames = input_dim[0]
    valid_length = start_f + speed_factor*frames < temp_images.shape[0]
    if valid_length:
        contain_division = ground_truth_convert.contain_division_check(sub_seq, ground_truth, frames=frames, speed_factor=1)
        contain_death = ground_truth_convert.contain_death_check(sub_seq, ground_truth, frames=frames, speed_factor=1)
        return (not contain_death) and (not contain_division)
    else:
        return valid_length


def death_sequence_valid_check(sub_seq, input_dim, annotations):
    """
    This function check if the given sub_seq is valid for speedup process, in terms of length, contain division and contain death
    :param sub_seq:
    :param input_dim:
    :param annotations:
    :return: new sub sequence
    """
    start_f = sub_seq['start_frame']
    frames = input_dim[0]
    death_frame = np.argmax(annotations['death'][sub_seq['cell'], start_f:start_f + frames])
    death_row = annotations['rows'][sub_seq['cell'], start_f + death_frame]
    death_col = annotations['cols'][sub_seq['cell'], start_f + death_frame]
    coordinate_correct = ground_truth_convert.ground_truth_scaling(death_row, death_col, sub_seq['offset_x'],
                                                                   sub_seq['offset_y'], input_dim, (None, *input_dim))
    sub_seq = ground_truth_convert.invalid_scope_check(coordinate_correct, death_frame, input_dim, sub_seq, annotations)

    return sub_seq


def generate_input_sequence(sub_seq, input_dim, temp_images, speed_factor=1):
    """
    This function generates the input image stack for model training.
    :param sub_seq: (dictionary) {'offset_x': int, 'offset_y': int, 'start_frame': int} indicates the starting point of
    the target cell.
    :param input_dim: (list) (frames, rows, cols, channels) indicates the input shape of the model.
    :param temp_images: (np_array) [frames, rows, cols, channels] the full scaled raw image array.
    :param speed_factor: (init) factor of speedup the input sequence with default value of 1. When the factor is 1, the
    input is at normal speed. A factor of 2 means the frequency of the input is 2 times than the normal one.
    :return: input_images: (np_array) [frames, rows, cols, channels] image stack for training without batch
    """

    start_f = sub_seq['start_frame']
    frames = input_dim[0]
    start_col = sub_seq['offset_x']
    cols = input_dim[2]
    start_row = sub_seq['offset_y']
    rows = input_dim[1]
    if input_dim[-1] == 3:
        input_images = temp_images[start_f:start_f + speed_factor * frames:speed_factor, start_row:start_row + rows,
                                   start_col:start_col + cols, :]
    if input_dim[-1] == 1:
        # when the number of channel is 1, we use CFP as input channel in HCT116 dataset, otherwise 0.
        input_images = temp_images[start_f:start_f + speed_factor*frames:speed_factor, start_row:start_row + rows,
                                   start_col:start_col + cols, 0]
        input_images = np.expand_dims(input_images, axis=-1)
        if input_images.shape[0] < frames:
            print('Padding the input using the last frame image')
            tiled_image = np.tile(input_images[-1, ], (frames-input_images.shape[0], 1, 1, 1))
            input_images = np.append(input_images, tiled_image, axis=0)
    return input_images


def generate_init_state(ground_truth, sub_seq, input_dim, output_shape, convlstm_loc, init_mask=False, roll_back=False, additionals={}):
    """
    This function generates the initial state for the ConvLSTM using corresponding ground truth. Because the target cell
    is guaranteed in scope at the first frame, the confidence map is used as a init mask, which multiple the first frame
    feature map. The confidence maps for the rest frames are all ones. The shape of this initial state should be the
    same with the shape of ConvLSTM input. The type of the generated map is square map by default.
    :param ground_truth: (dictionary) {'rows': list, 'cols': list} records the position of the target cell at all time
    points
    :param sub_seq: (dictionary) {'offset_x': int, 'offset_y': int, 'start_frame': int} indicates the starting point of
    the target cell
    :param input_dim: (list) (frames, rows, cols, channels) indicates the input shape of the model
    :param output_shape: (list) (batch, frames, rows, cols, channels) indicates the output shape of the model
    :return: init_state: (np_array) [frames, rows, cols, channels]
    """

    start_f = sub_seq['start_frame']
    if roll_back:
        # in roll back mode, the init state is the prediction from previous sequence
        true_col = sub_seq['first_col']
        true_row = sub_seq['first_row']
    else:
        true_col = ground_truth['cols'][start_f]
        true_row = ground_truth['rows'][start_f]


    # init_state = np.ones((*output_shape[1:-1], 3))

    if init_mask:
        # in init_mask mode, the init_state only has one channel, which indicate the initial location of the target cell
        coordinate_correct = ground_truth_convert.ground_truth_scaling(true_row, true_col, sub_seq['offset_x'],
                                                                       sub_seq['offset_y'], input_dim, (None, *input_dim))
        init_state = np.ones((*input_dim[1:-1], 1))
        init_state = np.repeat(np.expand_dims(
            ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, (None, *input_dim), type='square',
                                                               **additionals), axis=-1), init_state.shape[-1], axis=-1)
    else:
        # convlstm last version
        if convlstm_loc == 'conv_last':
            coordinate_correct = ground_truth_convert.ground_truth_scaling(true_row, true_col, sub_seq['offset_x'],
                                                                           sub_seq['offset_y'], input_dim, output_shape)
            init_state = np.zeros(output_shape[2::])
            init_state[:, :, 0] = ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, output_shape, type='square_single')
        elif convlstm_loc == 'conv_first':

            coordinate_correct = ground_truth_convert.ground_truth_scaling(true_row, true_col, sub_seq['offset_x'],
                                                                           sub_seq['offset_y'], input_dim, (None, *input_dim))
            init_state = np.zeros((*input_dim[1:-1], 16))
            init_state = np.repeat(np.expand_dims(
                ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, (None, *input_dim[0:-1], 16), type='square'), axis=-1),
                                               init_state.shape[-1], axis=-1)
        elif convlstm_loc == 'init_repeat':
            coordinate_correct = ground_truth_convert.ground_truth_scaling(true_row, true_col, sub_seq['offset_x'],
                                                                           sub_seq['offset_y'], input_dim,
                                                                           (None, *input_dim))
            init_state = np.zeros(input_dim)
            init_state[0,] = np.expand_dims(
                ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, (None, *input_dim),
                                                                   type='square'), axis=-1)
    return init_state


def generate_annotation_set(annotations, configuration, sub_seq_settings):
    cell_type = sub_seq_settings['cell_type']
    window_size = sub_seq_settings['window_size']
    window_step = sub_seq_settings['window_step']
    frame_size = sub_seq_settings['frame_size']
    frame_step = sub_seq_settings['frame_step']

    # iterate offset coordinates using sliding window
    offset_x_range = np.arange(0, configuration[cell_type]['experiments'][0]['cols'] - window_size + 1, window_step)
    offset_y_range = np.arange(0, configuration[cell_type]['experiments'][0]['rows'] - window_size + 1, window_step)
    offset_xs, offset_ys = np.meshgrid(offset_x_range, offset_y_range, sparse=False, indexing='xy')
    offset_xs = np.squeeze(np.reshape(offset_xs, (1, -1)))
    offset_ys = np.squeeze(np.reshape(offset_ys, (1, -1)))
    sub_seqs = []
    for offset_xy in zip(offset_xs, offset_ys):
        sub_seq_per_offset = ground_truth_convert.segment_cell_subsequence(annotations,
                                                                           configuration[cell_type]['experiments'],
                                                                           offset_xy[0], offset_xy[1], window_size,
                                                                           frame_size, frame_step)
        sub_seqs.extend(sub_seq_per_offset)
    # example
    # {'offset_x': 256, 'offset_y': 256, 'start_frame': 125, 'first_col': 338, 'first_row': 394, 'cell': 2, 'experiment': 76, 'target_at_begin': 1, 'include_division': 0, 'include_death': 0}

    sub_seqs_normal = ground_truth_convert.select_normal_subsequence(sub_seqs)
    sub_seqs_death = ground_truth_convert.select_death_subsequence(sub_seqs)
    sub_seqs_division = ground_truth_convert.select_division_subsequence(sub_seqs)
    print('normal: {}'.format(len(sub_seqs_normal)))
    print('death: {}'.format(len(sub_seqs_death)))
    print('division: {}'.format(len(sub_seqs_division)))
    return {'normal': sub_seqs_normal, 'death': sub_seqs_death, 'division': sub_seqs_division}


def sub_seqs_statistics(sub_seqs, sub_seq_settings):

    window_size = sub_seq_settings['window_size']
    window_step = sub_seq_settings['window_step']
    frame_size = sub_seq_settings['frame_size']
    frame_step = sub_seq_settings['frame_step']

    # print(len(sub_seqs))
    # example
    # {'offset_x': 256, 'offset_y': 256, 'start_frame': 125, 'first_col': 338, 'first_row': 394, 'cell': 2, 'experiment': 76, 'target_at_begin': 1, 'include_division': 0, 'include_death': 0}

    sub_seqs_normal = ground_truth_convert.count_normal_subsequence(sub_seqs)
    sub_seqs_death = ground_truth_convert.count_death_subsequence(sub_seqs)
    sub_seqs_division = ground_truth_convert.count_division_subsequence(sub_seqs)
    print('normal: {}'.format(sub_seqs_normal))
    print('death: {}'.format(sub_seqs_death))
    print('division: {}'.format(sub_seqs_division))


def annotations_check(cell_type, configuration=None):
    """
    This function check whether the original compressed annotation file exist
    :param cell_type: (string) the used cell type
    :param configuration: (dictionary) config information, including annotation path.
    :return: (bool) binary indicator for the validation of the annotation file
    """
    if configuration is None:
        configuration = read_configuration()
    annotation_path = configuration['annotation_path']
    file_path = '{}{}/annotations.pkl'.format(annotation_path, cell_type)
    return os.path.isfile(file_path)


def read_annotations(cell_type, configuration=None):
    """
    This function read the original compressed annotation
    :param cell_type: (string) the used cell type
    :param configuration: (dictionary) config information, including annotation path.
    :return: (dictionary) annotations
    """
    if configuration is None:
        configuration = read_configuration()
    annotation_path = configuration['annotation_path']
    file_path = '{}{}/annotations.pkl'.format(annotation_path, cell_type)
    with open(file_path, 'rb') as f:
        annotations = pickle.load(f)
    if 'cell_type' not in annotations:
        annotations.update({'cell_type': cell_type})
    return annotations


def correct_annotations(annotations, cell_type):
    """
    This function read the original compressed annotation
    :param annotations: (dictionary) the used cell type
    :param cell_type: (string) the used cell type
    :return: (dictionary) new annotations
    """
    new_annotations = ground_truth_convert.correct_division_annotations(annotations, cell_type)
    # new_annotations = ground_truth_convert.correct_death_annotations(new_annotations, cell_type)
    return new_annotations


def select_annotation_for_subseq(annotations, sub_seq):
    """
    This function select the specific row of the annotations based on the cell id in sub_seq.
    :param annotations:
    :param sub_seq:
    :return: (dictionary) ['rows': list, 'cols': list] coordinates for the same cell
    """
    cell = sub_seq['cell']
    ground_truth = {'rows': annotations['rows'][cell, :],
                    'cols': annotations['cols'][cell, :],
                    'death': annotations['death'][cell, :],
                    'division': annotations['division'][cell, :]}
    return ground_truth


def annotation_set_check(configuration, sub_seq_settings):
    annotation_path = '{}{}/'.format(configuration['annotation_path'], sub_seq_settings['cell_type'])
    file_name = '{}.pkl'.format(ground_truth_convert.get_training_name(sub_seq_settings))
    file_path = '{}annotation_sets/{}'.format(annotation_path, file_name)
    return os.path.isfile(file_path)


def save_annotation_set(annotation_set, configuration, sub_seq_settings):
    annotation_path = '{}{}/'.format(configuration['annotation_path'], sub_seq_settings['cell_type'])
    # save the annotation set as file, first check the directory
    if not os.path.isdir(annotation_path):
        try:
            os.mkdir(annotation_path)
        except OSError:
            print("Creation of the directory {} failed".format(annotation_path))
        else:
            print("Successfully created the directory {} ".format(annotation_path))

    if not os.path.isdir('{}annotation_sets/'.format(annotation_path)):
        try:
            os.mkdir('{}annotation_sets/'.format(annotation_path))
        except OSError:
            print("Creation of the directory {} failed".format('{}annotation_sets/'.format(annotation_path)))
        else:
            print("Successfully created the directory {} ".format('{}annotation_sets/'.format(annotation_path)))

    file_name = '{}.pkl'.format(ground_truth_convert.get_training_name(sub_seq_settings))
    file_path = '{}annotation_sets/{}'.format(annotation_path, file_name)
    output = open(file_path, 'wb')
    pickle.dump(annotation_set, output)
    output.close()


def read_annotation_set(configuration, sub_seq_settings):
    annotation_path = '{}{}/'.format(configuration['annotation_path'], sub_seq_settings['cell_type'])
    file_name = '{}.pkl'.format(ground_truth_convert.get_training_name(sub_seq_settings))
    file_path = '{}annotation_sets/{}'.format(annotation_path, file_name)
    with open(file_path, 'rb') as f:
        annotation_set = pickle.load(f)
    return annotation_set


def generate_training_testing_subseqs(annotation_set, testing_exp=[78], mode='normal', sampling_size=[0, 0]):
    if mode == 'normal':
        print('Selecting normal subseqs only...')
        sub_seqs_normal = annotation_set['normal']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal, testing_exp)
    elif mode == 'death':
        # In the death mode, the dataset only contains death subseqs
        print('Selecting death subseqs...')
        sub_seqs_death = annotation_set['death']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_death, testing_exp)
    elif mode == 'death_balanced':
        # In the death_balanced mode, the dataset mixs death and normal subseqs together
        print('Selecting death subseqs...')
        sub_seqs_death = annotation_set['death']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_death, testing_exp)
        print('Selecting normal subseqs...')
        sub_seqs_normal = annotation_set['normal']
        testing_subseqs_normal, training_subseqs_normal = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal,
                                                                                                   testing_exp)
        print('Blending death and normal subseqs...')
        training_subseqs.extend(random.sample(training_subseqs_normal,
                                              min(len(training_subseqs), len(training_subseqs_normal))))
        testing_subseqs.extend(random.sample(testing_subseqs_normal,
                                             min(len(testing_subseqs), len(testing_subseqs_normal))))
    elif mode == 'division':
        print('Selecting division subseqs...')
        sub_seqs_normal = annotation_set['division']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal, testing_exp)
    elif mode == 'division_balanced':
        # In the division_balanced mode, the dataset mixs division and normal subseqs together
        print('Selecting division subseqs...')
        sub_seqs_normal = annotation_set['division']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal, testing_exp)
        print('Selecting normal subseqs...')
        sub_seqs_normal = annotation_set['normal']
        testing_subseqs_normal, training_subseqs_normal = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal,
                                                                                                   testing_exp)
        print('Blending division and normal subseqs...')
        training_subseqs.extend(random.sample(training_subseqs_normal,
                                              min(len(training_subseqs), len(training_subseqs_normal))))
        testing_subseqs.extend(random.sample(testing_subseqs_normal,
                                             min(len(testing_subseqs), len(testing_subseqs_normal))))
    elif mode == 'all':
        print('Selecting all types of subseqs only...')
        sub_seqs_normal = annotation_set['normal']
        testing_subseqs, training_subseqs = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal,
                                                                                                   testing_exp)

        sub_seqs_death = annotation_set['death']
        testing_subseqs_death, training_subseqs_death = ground_truth_convert.split_subseq_by_exp(sub_seqs_death,
                                                                                                   testing_exp)
        testing_subseqs.extend(testing_subseqs_death)
        training_subseqs.extend(training_subseqs_death)

        sub_seqs_division = annotation_set['division']
        testing_subseqs_division, training_subseqs_division = ground_truth_convert.split_subseq_by_exp(sub_seqs_division,
                                                                                                   testing_exp)
        testing_subseqs.extend(testing_subseqs_division)
        training_subseqs.extend(training_subseqs_division)
    elif mode == 'balanced':
        print('Selecting all types of subseqs...')

        testing_subseqs = []
        training_subseqs = []

        sub_seqs_death = annotation_set['death']
        testing_subseqs_death, training_subseqs_death = ground_truth_convert.split_subseq_by_exp(sub_seqs_death,
                                                                                                   testing_exp)
        testing_subseqs.extend(random.sample(testing_subseqs_death, min(round(sampling_size[0]/3), len(testing_subseqs_death))))
        training_subseqs.extend(random.sample(training_subseqs_death, min(round(sampling_size[1] / 3), len(training_subseqs_death))))

        sub_seqs_division = annotation_set['division']
        testing_subseqs_division, training_subseqs_division = ground_truth_convert.split_subseq_by_exp(sub_seqs_division,
                                                                                                   testing_exp)
        testing_subseqs.extend(random.sample(testing_subseqs_division,
                                             min(round((sampling_size[0] - len(testing_subseqs))/2),
                                                 len(testing_subseqs_division))))
        training_subseqs.extend(random.sample(training_subseqs_division,
                                              min(round((sampling_size[1] - len(training_subseqs))/2),
                                                  len(training_subseqs_division))))

        sub_seqs_normal = annotation_set['normal']
        testing_subseqs_normal, training_subseqs_normal = ground_truth_convert.split_subseq_by_exp(sub_seqs_normal,
                                                                                                   testing_exp)
        testing_subseqs.extend(random.sample(testing_subseqs_normal,
                                             min(sampling_size[0] - len(testing_subseqs),
                                                 len(testing_subseqs_normal))))
        training_subseqs.extend(random.sample(training_subseqs_normal,
                                              min(sampling_size[1] - len(training_subseqs),
                                                  len(training_subseqs_normal))))

    return training_subseqs, testing_subseqs


def read_image_array(configuration, sub_seq):
    if 'cell_type' not in sub_seq:
        sub_seq.update({'cell_type': 'HCT116'})
    target_exp = ground_truth_convert.select_config_by_exp(configuration, sub_seq['cell_type'], sub_seq['experiment'])
    temp_image_file = '{}{}/Pos{}_c{}f{}.npy'.format(configuration['temp_image_path'], sub_seq['cell_type'],
                                                     sub_seq['experiment'], target_exp['channels'],
                                                     target_exp['frames'])
    temp_images = np.load(temp_image_file)
    return temp_images


def generate_hard_evaluation_predictions(annotations, testing_subseqs, input_dim, output_shape, additionals):
    return evaluate_utils.generate_hard_evaluation_predictions(annotations, testing_subseqs, input_dim, output_shape, additionals)


def evaluation_predictions_check(sampling_size, model_type, restore_pix):
    return evaluate_utils.evaluation_predictions_check(sampling_size, model_type, restore_pix)


def save_evaluation_predictions(prediction, sampling_size, model_type, restore_pix):
    evaluate_utils.save_evaluation_predictions(prediction, sampling_size, model_type, restore_pix)


def read_evaluation_predictions(sampling_size, model_type, restore_pix):
    return evaluate_utils.read_evaluation_predictions(sampling_size, model_type, restore_pix)


def model_evaluation(predict, annotations, testing_subseqs, input_dim, output_shape):
    return evaluate_utils.model_evaluation(predict, annotations, testing_subseqs, input_dim, output_shape)


def multitask_model_evaluation (predict, annotations, testing_subseqs, input_dim, output_shapes):
    return evaluate_utils.multitask_model_evaluation(predict, annotations, testing_subseqs, input_dim, output_shapes)


def evaluation_results_check(model_type, restore_pix):
    return evaluate_utils.evaluation_results_check(model_type, restore_pix)


def save_evaluation_results(df, model_type, restore_pix):
    evaluate_utils.save_evaluation_results(df, model_type, restore_pix)


def read_evaluation_results(model_type, restore_pix):
    return evaluate_utils.read_evaluation_results(model_type, restore_pix)



