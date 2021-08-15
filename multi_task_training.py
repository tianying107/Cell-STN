import utility as ut
import models
from lineage_classes import DataGenerator, sorted_binary_crossentropy, adjusted_RMSE, adjusted_peak_accuracy, MultitaskDataGenerator, event_peak_accuracy, event_sequence_accuracy
import os
import sys
minor_version = sys.version_info[1]
if minor_version == 5:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    # from keras.metrics import Recall, Precision
elif minor_version == 6:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    # from tensorflow.keras.metrics import Recall, Precision
import random


# read configuration
configuration = ut.read_configuration()
print('production settings: {}'.format(configuration['production']))
minor_version = sys.version_info[1]
print('Current python minor version: {}'.format(minor_version))

# setup parameters
training_sampling_size = 3500
testing_sampling_size = 600
# loss = weighted_binary_crossentropy([1, 10]) # default value for the first 10 epoch
if ut.deploy_check():
    model_type = sys.argv[1]
    workers = 6
    command_dict = ut.alloc_settings_command(sys.argv)
    cell_type, input_channel, epochs, batch_size, pix, train_mode, restore_pix, loss, gt_paras, speedup, gss, blender = ut.get_settings(
        command_dict)
else:
    commands = ['current_file', 'convlstm_v785', '-c', 3, '-batch', 1, '-cell', 'U2OS', '-blender', 'True']
    model_type = commands[1]
    workers = 1
    command_dict = ut.alloc_settings_command(commands)
    cell_type, input_channel, epochs, batch_size, pix, train_mode, restore_pix, loss, gt_paras, speedup, gss, blender = ut.get_settings(
        command_dict)
    training_sampling_size = 4
    testing_sampling_size = 4

frame_size = 10  # default value
check_period = 1  # default value
patience = 5  # default value

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

# annotation correction
annotations = ut.correct_annotations(annotations, cell_type)

# setup model
model, model_note = models.create_lineage_model(model_type, deploy_mode=ut.deploy_check(), input_channel=input_channel,
                                                cell_type=cell_type)
tasks = [False, False, False]
loss = {}
metrics = {}
output_shapes = {}
for output in model.outputs:
    output_name = output.name.split('/')[0]
    if output_name == 'tracking_output_layer':
        loss.update({'tracking_output_layer': 'binary_crossentropy'})
        metrics.update({'tracking_output_layer': adjusted_peak_accuracy})
        output_shapes.update({'tracking': output.get_shape().as_list()})
        tasks[0] = True
    elif output_name == 'death_output_layer':
        loss.update({'death_output_layer': 'binary_crossentropy'})
        if blender:
            metrics.update({'death_output_layer': [event_peak_accuracy, event_sequence_accuracy]})
        else:
            metrics.update({'death_output_layer': event_peak_accuracy})
        output_shapes.update({'death': output.get_shape().as_list()})
        tasks[1] = True
    elif output_name == 'division_output_layer':
        if output.get_shape().as_list()[-1] == 3:
            loss.update({'division_output_layer': sorted_binary_crossentropy()})
        else:
            loss.update({'division_output_layer': 'binary_crossentropy'})
        if blender:
            metrics.update({'division_output_layer': [event_peak_accuracy, event_sequence_accuracy]})
        else:
            metrics.update({'division_output_layer': event_peak_accuracy})
        output_shapes.update({'division': output.get_shape().as_list()})
        tasks[2] = True

# split subseqs for training and testing
if tasks[1] & (not tasks[2]):
    if blender:
        training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='death_balanced',
                                                                             testing_exp=configuration[cell_type][
                                                                                 'validation_sets'])
        # import pandas as pd
        # df = pd.DataFrame(training_subseqs)
        # df = df[(df.cell == 1) | (df.cell == 2) | (df.cell == 15) | (df.cell == 19) | (df.cell == 10) | (df.cell == 110) | (df.cell == 21) | (df.cell == 24)]
        # exclude_seq = df.to_dict(orient='record')
        # training_subseqs = [seq for seq in training_subseqs if seq not in exclude_seq]
        # testing_subseqs.extend(exclude_seq)
        #
        # df = pd.DataFrame(testing_subseqs)
        # df = df[(df.cell == 169) | (df.cell == 156) | (df.cell == 152) | (df.cell == 164)]
        # exclude_seq_2 = df.to_dict(orient='record')
        # testing_subseqs = [seq for seq in testing_subseqs if seq not in exclude_seq_2]
        # training_subseqs.extend(exclude_seq_2)
    else:
        training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='death',
                                                                                 testing_exp=configuration[cell_type][
                                                                                     'validation_sets'])
elif tasks[2] & (not tasks[0]) & (not tasks[1]):
    if blender:
        training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='division_balanced',
                                                                             testing_exp=configuration[cell_type][
                                                                                 'validation_sets'])
    else:
        training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='division',
                                                                                 testing_exp=configuration[cell_type][
                                                                                     'validation_sets'])
else:
    training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='balanced', sampling_size=[testing_sampling_size, training_sampling_size], testing_exp=configuration[cell_type]['validation_sets'])

print('Total training samples: {}'.format(len(training_subseqs)))
ut.sub_seqs_statistics(training_subseqs, sub_seq_settings)
print('Total testing samples: {}'.format(len(testing_subseqs)))
ut.sub_seqs_statistics(testing_subseqs, sub_seq_settings)

input_dim = (sub_seq_settings['frame_size'], window_size, window_size, input_channel)


# setup optimizer and loss function
# optimizer = 'rmsprop'
# optimizer = optimizers.RMSprop(lr=0.0008)
optimizer = 'adam'
# loss = 'categorical_crossentropy'



# output_shapes = [output.get_shape().as_list() for out in model.outputs]
#
print(output_shapes)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)
model.summary()

if train_mode == 'continue':
    if os.path.isfile('{}{}{}.h5'.format(result_path, model_type, restore_pix)):
        print('restoring from model {}{}.h5'.format(model_type, restore_pix))
        model.load_weights('{}{}{}.h5'.format(result_path, model_type, restore_pix))
# exit()
# Parameters
params = {'input_dim': input_dim,
          'batch_size': batch_size,
          'shuffle': True,
          'rotate': True,
          'model_note': model_note,
          'gt_paras': gt_paras}
# Generators
if gss:
    training_generator = MultitaskDataGenerator(training_subseqs, annotations, output_shapes, **params,
                                                use_sample=training_sampling_size, speedup=speedup)
else:
    training_generator = MultitaskDataGenerator(training_subseqs, annotations, output_shapes, **params, speedup=speedup)
validation_generator = MultitaskDataGenerator(testing_subseqs, annotations, output_shapes, **params, speedup=speedup)

# checkpoint = ModelCheckpoint('{}{}{}_check.h5'.format(result_path, model_type, pix),
#                              monitor='val_loss',
#                              save_best_only=True,
#                              mode='min',
#                              period=check_period)
checkpoint = ModelCheckpoint('{}{}{}_check.h5'.format(result_path, model_type, pix),
                             monitor='val_event_sequence_accuracy',
                             save_best_only=True,
                             mode='max',
                             period=check_period)
tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/')) if model_type + pix in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/') + model_type + pix + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=[checkpoint, tensorboard],
                    use_multiprocessing=True,
                    workers=workers)
# callbacks=[checkpoint, tensorboard],
model.save_weights('{}{}{}.h5'.format(result_path, model_type, pix))
