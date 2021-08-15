import utility as ut
import models
from lineage_classes import DataGenerator, weighted_binary_crossentropy, sequence_binary_crossentropy, adjusted_RMSE, adjusted_peak_accuracy
import os
import sys
minor_version = sys.version_info[1]
if minor_version == 5:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
elif minor_version == 6:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import random


# read configuration
configuration = ut.read_configuration()
print('production settings: {}'.format(configuration['production']))
minor_version = sys.version_info[1]
print('Current python minor version: {}'.format(minor_version))

# setup parameters
training_sampling_size = 3500
testing_sampling_size = 600
if ut.deploy_check():
    model_type = sys.argv[1]
    workers = 6
    command_dict = ut.alloc_settings_command(sys.argv)
    cell_type, input_channel, epochs, batch_size, pix, train_mode, restore_pix, loss, gt_paras, speedup, gss, _ = ut.get_settings(
        command_dict)
    if gss:
        training_sampling_size = 3000
else:
    commands = ['current_file', 'convlstm_v78', '-c', 3, '-batch', 1, '-gss', 'True', '-cell', 'U2OS']
    model_type = commands[1]
    workers = 1
    command_dict = ut.alloc_settings_command(commands)
    cell_type, input_channel, epochs, batch_size, pix, train_mode, restore_pix, loss, gt_paras, speedup, gss, _ = ut.get_settings(
        command_dict)
    training_sampling_size = 2
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

# split subseqs for training and testing
training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, testing_exp=configuration[cell_type]['validation_sets'])
if not gss:
    # if GSS is not enables, use partial of the training set by sampling
    training_subseqs = random.sample(training_subseqs,min(training_sampling_size, len(training_subseqs)))
testing_subseqs = random.sample(testing_subseqs,min(testing_sampling_size, len(testing_subseqs)))
print('Number of training samples: {}'.format(len(training_subseqs)))
print('Number of testing samples: {}'.format(len(testing_subseqs)))

input_dim = (sub_seq_settings['frame_size'], window_size, window_size, input_channel)

# setup model
model, convlstm_loc = models.create_lineage_model(model_type, deploy_mode=ut.deploy_check(), input_channel=input_channel)

# setup optimizer and loss function
# optimizer = 'rmsprop'
optimizer = 'adam'
# loss = 'categorical_crossentropy'

if train_mode == 'continue':
    if os.path.isfile('{}{}{}.h5'.format(result_path, model_type, restore_pix)):
        print('restoring from model {}{}.h5'.format(model_type, restore_pix))
        model.load_weights('{}{}{}.h5'.format(result_path, model_type, restore_pix))

output_shape = model.layers[-1].output_shape
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[adjusted_RMSE, adjusted_peak_accuracy])
model.summary()

# Parameters
params = {'input_dim': input_dim,
          'batch_size': batch_size,
          'shuffle': True,
          'rotate': True,
          'convlstm_loc': convlstm_loc,
          'gt_paras': gt_paras}
# Generators
if gss:
    training_generator = DataGenerator(training_subseqs, annotations, output_shape, **params,
                                       use_sample=training_sampling_size, speedup=speedup)
else:
    training_generator = DataGenerator(training_subseqs, annotations, output_shape, **params, speedup=speedup)
validation_generator = DataGenerator(testing_subseqs, annotations, output_shape, **params)


early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=patience,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint('{}{}{}_check.h5'.format(result_path, model_type, pix),
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=check_period)
tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/')) if model_type + pix in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/') + model_type + pix + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=[tensorboard, checkpoint],
                    use_multiprocessing=True,
                    workers=workers)
# callbacks=[checkpoint, tensorboard],
model.save_weights('{}{}{}.h5'.format(result_path, model_type, pix))


# model evaluation
_, evaluation_subseqs = ut.generate_training_testing_subseqs(annotation_set, testing_exp=configuration[cell_type]['validation_sets'])
random.seed(configuration[cell_type]['evaluation_settings']['seed'])
sampling_size = configuration[cell_type]['evaluation_settings']['sampling_size']
evaluation_subseqs = random.sample(evaluation_subseqs, min(sampling_size, len(evaluation_subseqs)))
# Parameters
evaluate_params = {'input_dim': input_dim,
                   'batch_size': batch_size,
                   'shuffle': False,
                   'rotate': False,
                   'convlstm_loc': convlstm_loc}
# Generators
evaluation_generator = DataGenerator(evaluation_subseqs, annotations, output_shape, **evaluate_params)

# model evaluation
prediction = model.predict_generator(generator=evaluation_generator,
                                     use_multiprocessing=True,
                                     workers=workers)
ut.save_evaluation_predictions(prediction, sampling_size, model_type, pix)
df, peaks = ut.model_evaluation(prediction, annotations, evaluation_subseqs, input_dim, output_shape)
ut.save_evaluation_results(df, model_type, restore_pix)
