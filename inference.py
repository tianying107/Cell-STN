import pylab as plt
from utilities import ground_truth_convert
import utility as ut
import sys
import models
import os
import numpy as np
from utilities import coordinate2map as c2m
from random import sample
import pandas as pd

# tensorboard --logdir=/Users/stn/convlstm_v4_35-c1_1
# read configuration
configuration = ut.read_configuration()
print('production settings: {}'.format(configuration['production']))
cell_type = 'HCT116'
if ut.deploy_check():
    model_type = sys.argv[1]
    workers = 8
    batch_size = 16  # default value
else:
    model_type = 'convlstm_v78'
    restore_pix = '-c3-i2-gss-con-1_check'  # -c3-i2-gss-con-1_check   -c1-2-wpc  -c1-con-4-wac  -c1-4 -c1-i2  -c3-i2-con-1_check -c3-gss-1_check -c3-i10-gss-MCF7-con-1_check
    workers = 4
    batch_size = 2  # default value
input_channel = 3
pix = ''
frame_size = 10  # default value
window_size = 256
result_path = './benchmarks/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# main begin
if not ut.temp_dataset_check(cell_type):
    print('No temporal dataset found.')
    print('Generating temporal data...')
    ut.generate_temp_data(cell_type)

# read ground-truth annotations
if not ut.annotations_check(cell_type):
    raise NameError('The annotation file does not exist. Please check the file in ./Annotation or run the '
                    'script compress_annotations.py to generate.')
annotations = ut.read_annotations(cell_type)

# setup sub-sequence set
sub_seq_settings = {'cell_type': cell_type, 'window_size': window_size, 'window_step': 128, 'frame_size': frame_size, 'frame_step': 5}

# Check converted annotation set
if ut.annotation_set_check(configuration, sub_seq_settings):
    print('Reading subseqs...')
    annotation_set = ut.read_annotation_set(configuration, sub_seq_settings)
else:
    # save subseq as file
    print('Generating subseqs...')
    annotation_set = ut.generate_annotation_set(annotations, configuration, sub_seq_settings)
    print('Saving subseqs...')
    ut.save_annotation_set(annotation_set, configuration, sub_seq_settings)

# split subseqs for training and testing
training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, testing_exp=configuration[cell_type]['validation_sets'])
training_sampling_size = 3500
testing_sampling_size = 600
# training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='death', sampling_size=[testing_sampling_size, training_sampling_size])
print('Number of training samples: {}'.format(len(training_subseqs)))
print('Number of testing samples: {}'.format(len(testing_subseqs)))
ut.sub_seqs_statistics(testing_subseqs, sub_seq_settings)

# exit()
input_dim = (sub_seq_settings['frame_size'], window_size, window_size, input_channel)


model, convlstm_loc = models.create_lineage_model(model_type, deploy_mode=ut.deploy_check(), input_channel=input_channel)
model.summary()
# model = Model(inputs=model.inputs, outputs=[model.output,model.get_layer('ev3_norm').output])

# best: c1-con-2-bce
model.load_weights('./benchmarks/{}{}.h5'.format(model_type, restore_pix))

sub_seq = testing_subseqs[60]
# df = pd.DataFrame(testing_subseqs)
# df = df[(df.cell == 160) & (df.start_frame == 5) & (df.offset_x == 384) & (df.offset_y == 384)]
# sub_seq = df.to_dict(orient='record')[0]
# exit()
# 50, 145, 135, 100, 60, 40
# ############
# sub_seq = {'cell': 191, 'experiment': 78, 'first_col': 527, 'first_row': 476, 'include_death': 0, 'include_division': 0, 'last_col': 517, 'last_row': 473, 'offset_x': 384, 'offset_y': 256, 'start_frame': 30, 'target_at_begin': 1}
# sub_seq = {'cell': 178, 'experiment': 78, 'first_col': 500, 'first_row': 501, 'include_death': 0, 'include_division': 0, 'last_col': 461, 'last_row': 488, 'offset_x': 256, 'offset_y': 256, 'start_frame': 30, 'target_at_begin': 1}
# sub_seq = {'cell': 157, 'experiment': 78, 'first_col': 536, 'first_row': 513, 'include_death': 0, 'include_division': 0, 'last_col': 521, 'last_row': 528, 'offset_x': 384, 'offset_y': 384, 'start_frame': 25, 'target_at_begin': 1}
# sub_seq = {'cell': 152, 'experiment': 78, 'first_col': 516, 'first_row': 451, 'include_death': 0, 'include_division': 0, 'last_col': 517, 'last_row': 445, 'offset_x': 384, 'offset_y': 384, 'start_frame': 25, 'target_at_begin': 1}
print(sub_seq)

roll_back = False

if not roll_back:
    # normal mode

    # sub_seq['start_frame'] += 3
    # sub_seq['first_col'] = 123 + sub_seq['offset_x'] + 1
    # sub_seq['first_row'] = 161 + sub_seq['offset_y'] + 1

    # sub_seq['start_frame'] += 6
    # sub_seq['first_col'] = 132 + sub_seq['offset_x'] + 1
    # sub_seq['first_row'] = 161 + sub_seq['offset_y'] + 1

    temp_images = ut.read_image_array(configuration, sub_seq)

    input_seq = ut.generate_input_sequence(sub_seq, input_dim, temp_images)
    input_seq = np.expand_dims(input_seq, axis=0)

    ground_truth = ut.select_annotation_for_subseq(annotations, sub_seq)
    output_shape = model.get_layer('tracking_output_layer').output_shape

    gt = ut.generate_ground_truth_sequence(ground_truth, sub_seq, input_dim, output_shape)

    init_mask = ((convlstm_loc == 'init_mask') | (convlstm_loc == 'init_state_separate'))

    init_state = ut.generate_init_state(ground_truth, sub_seq, input_dim, output_shape, convlstm_loc,
                                        init_mask=init_mask, roll_back=True)
    init_state = np.expand_dims(init_state, axis=0)

    results = model.predict([input_seq, init_state])
    # # death session
    # print(results[1])
    # death_output_shape = model.get_layer('death_output_layer').output_shape
    # gt_death = ut.generate_death_truth_sequence(ground_truth, sub_seq, death_output_shape)
    # print('death gt: {}'.format(gt_death))
    # results, middle = model.predict([input_seq, init_state])
    results = np.squeeze(results[0])
else:

    # roll back evaluation
    temp_sub_seq = sub_seq.copy()
    results = np.zeros((frame_size, window_size, window_size))
    predictions = np.zeros((frame_size, 2))
    for iteration in range(3):
        temp_sub_seq['start_frame'] = sub_seq['start_frame'] + iteration * 3
        # temp_sub_seq['start_frame'] += iteration * 3
        if iteration > 0:
            temp_sub_seq['first_col'] = predictions[iteration * 3, 1] + temp_sub_seq['offset_x'] + 1
            temp_sub_seq['first_row'] = predictions[iteration * 3, 0] + temp_sub_seq['offset_y'] + 1

        temp_images = ut.read_image_array(configuration, temp_sub_seq)

        input_seq = ut.generate_input_sequence(temp_sub_seq, input_dim, temp_images)
        input_seq = np.expand_dims(input_seq, axis=0)

        ground_truth = ut.select_annotation_for_subseq(annotations, temp_sub_seq)
        output_shape = model.layers[-1].output_shape
        gt = ut.generate_ground_truth_sequence(ground_truth, temp_sub_seq, input_dim, output_shape)

        # if iteration == 0:
        init_mask = ((convlstm_loc == 'init_mask') | (convlstm_loc == 'init_state_separate'))

        init_state = ut.generate_init_state(ground_truth, temp_sub_seq, input_dim, output_shape, convlstm_loc,
                                            init_mask=init_mask, roll_back=True)
        init_state = np.expand_dims(init_state, axis=0)
        # else:
        #     coordinate_correct = ground_truth_convert.ground_truth_scaling(temp_sub_seq['first_row'], temp_sub_seq['first_col'], sub_seq['offset_x'],
        #                                                                    sub_seq['offset_y'], input_dim, output_shape)
        #     gaussian_filter = c2m.xy2gaussian(coordinate_correct[1], coordinate_correct[0], 256, 256)
        #     init_state = np.expand_dims(np.expand_dims(np.multiply(results[iteration * 3], gaussian_filter), axis=-1), axis=0)



        result = model.predict([input_seq, init_state])
        # results, middle = model.predict([input_seq, init_state])
        result = np.squeeze(result)

        pred_cols = np.argmax(np.max(result, axis=1), axis=1)
        pred_rows = np.argmax(np.max(result, axis=2), axis=1)
        pred = np.transpose(np.array((pred_rows[0:4], pred_cols[0:4])))
        predictions[iteration*3: iteration*3 + 4, :] = pred
        if iteration == 0:
            results[iteration * 3: iteration * 3 + 4, ] = result[0:4, ]
        else:
            results[iteration*3 + 1: iteration*3 + 4, ] = result[1:4, ]
        print(predictions)
    temp_images = ut.read_image_array(configuration, sub_seq)
    input_seq = ut.generate_input_sequence(sub_seq, input_dim, temp_images)
    input_seq = np.expand_dims(input_seq, axis=0)
print(sub_seq)
start = sub_seq['start_frame']
rows = ground_truth['rows'][start:start+frame_size]
cols = ground_truth['cols'][start:start+frame_size]

sequence = np.zeros(output_shape[1::])

fig, axs = plt.subplots(4, frame_size//2)
distance = 0

first_correct = ground_truth_convert.ground_truth_scaling(rows[0], cols[0],
                                                                   sub_seq['offset_x'], sub_seq['offset_y'],
                                                                   input_dim, output_shape)
# print(first_correct)
zoom_size = 256
zoom_row = int(max(min((first_correct[0]-zoom_size/2), window_size-zoom_size), 0))
zoom_col = int(max(min((first_correct[1]-zoom_size/2), window_size-zoom_size), 0))

input_seq[:,:,:,:,1] *=2
for f, coordinate in enumerate(zip(rows, cols)):
    print('frame: {}'.format(f))
    coordinate_correct = ground_truth_convert.ground_truth_scaling(coordinate[0], coordinate[1],
                                                                   sub_seq['offset_x'], sub_seq['offset_y'],
                                                                   input_dim, output_shape)
    print('gt: {}'.format(coordinate_correct))
    print('max: {}'.format(np.amax(results[f,])))
    pre_loc = np.where(results == np.amax(results[f,]))
    listOfCordinates = list(zip(pre_loc[1], pre_loc[2]))
    print('prediction: {}'.format(listOfCordinates))

    if input_channel == 1:
        display_channel = 0
    else:
        display_channel = 1

    axs[f//5][f%5].imshow(2*input_seq[0, f, zoom_row:zoom_row+zoom_size, zoom_col:zoom_col+zoom_size, :], cmap='gray')
    axs[f//5][f%5].plot(coordinate_correct[1]-zoom_col, coordinate_correct[0]-zoom_row, 'ro', markersize=10)
    axs[f//5][f%5].plot(listOfCordinates[0][1]-zoom_col, listOfCordinates[0][0]-zoom_row, 'bs', markersize=6)
    dis = np.sqrt(2*np.mean((np.array(coordinate_correct) - np.array(listOfCordinates[0]))**2))
    print('distance: {}'.format(dis))
    distance += dis

    axs[2+f//5][f%5].imshow(results[f, zoom_row:zoom_row+zoom_size, zoom_col:zoom_col+zoom_size], cmap='jet')
    # axs[2+f//5][f%5].plot(listOfCordinates[0][1]-zoom_col, listOfCordinates[0][0]-zoom_row, 'bx')
    # axs[2+f//5][f%5].plot(coordinate_correct[1]-zoom_col, coordinate_correct[0]-zoom_row, 'r+')

    # print(middle.shape)
    # axs[2][f].imshow(middle[0, f, :, :, 0])
    # axs[2][f].plot(coordinate_correct[1], coordinate_correct[0], 'r+')
    # axs[2+f//5][f%5].imshow(gt[f, :, :, 0])


print('Averaged distance: {}'.format(distance/frame_size))

plt.show()