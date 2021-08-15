import utility as ut
import sys
import models
import os
from lineage_classes import MultitaskDataGenerator, weighted_binary_crossentropy, RMSE, peak_accuracy, adjusted_peak_accuracy, adjusted_RMSE, event_peak_accuracy
import random


# read configuration
configuration = ut.read_configuration()
print('production settings: {}'.format(configuration['production']))
minor_version = sys.version_info[1]
print('Current python minor version: {}'.format(minor_version))

# setup parameters
roll_back = False
input_channel = None  # default value
train_mode = 'continue'  # default value
restore_pix = ''
loss = 'binary_crossentropy' # default value
# loss = weighted_binary_crossentropy([1, 10]) # default value for the first 10 epoch
gt_paras = {}
cell_type = 'HCT116'
if ut.deploy_check():
    model_type = sys.argv[1]
    restore_pix = sys.argv[2]

    workers = 6
    batch_size = 2  # default value
    input_channel = 1
    if len(sys.argv) > 3:
        input_channel = int(sys.argv[3])
    if len(sys.argv) > 4:
        cell_type = sys.argv[4]

else:
    model_type = 'convlstm_v785'
    workers = 2
    batch_size = 2  # default value
    restore_pix = '-c3-MCF7-bl-1_check'  # -c1-con-2-bce  hard_baseline -c1-con-3-3
    # roll_back = True
    input_channel = 3
    cell_type = 'MCF7'

frame_size = 10  # default value
random.seed(configuration[cell_type]['multitask_evaluation_settings']['seed'])
sampling_size = configuration[cell_type]['multitask_evaluation_settings']['sampling_size']
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
model, model_note = models.create_lineage_model(model_type, deploy_mode=ut.deploy_check(), input_channel=input_channel, cell_type=cell_type)
tasks = [False, False, False]
output_shapes = {}
loss = {}
metrics = {}
for output in model.outputs:
    output_name = output.name.split('/')[0]
    if output_name == 'tracking_output_layer':
        output_shapes.update({'tracking': output.get_shape().as_list()})
        tasks[0] = True
    elif output_name == 'death_output_layer':
        loss.update({'death_output_layer': 'binary_crossentropy'})
        metrics.update({'death_output_layer': event_peak_accuracy})
        output_shapes.update({'death': output.get_shape().as_list()})
        tasks[1] = True
    elif output_name == 'division_output_layer':
        loss.update({'division_output_layer': 'binary_crossentropy'})
        metrics.update({'division_output_layer': event_peak_accuracy})
        output_shapes.update({'division': output.get_shape().as_list()})
        tasks[2] = True

# split subseqs for training and testing
if tasks[1] & (not tasks[0]) & (not tasks[2]):
    # _, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='death', testing_exp=configuration[cell_type]['validation_sets'])
    training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='death_balanced',
                                                              testing_exp=configuration[cell_type]['validation_sets'])
    import pandas as pd

    df = pd.DataFrame(training_subseqs)
    df = df[(df.cell == 1) | (df.cell == 2) | (df.cell == 15) | (df.cell == 19) | (df.cell == 10) | (df.cell == 110)]
    exclude_seq = df.to_dict(orient='record')
    training_subseqs = [seq for seq in training_subseqs if seq not in exclude_seq]
    testing_subseqs.extend(exclude_seq)

    df = pd.DataFrame(testing_subseqs)
    df = df[(df.cell == 169) | (df.cell == 156) | (df.cell == 152) | (df.cell == 164)]
    exclude_seq_2 = df.to_dict(orient='record')
    testing_subseqs = [seq for seq in testing_subseqs if seq not in exclude_seq_2]
    training_subseqs.extend(exclude_seq_2)
elif tasks[2] & (not tasks[0]) & (not tasks[1]):
    # _, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='division', testing_exp=configuration[cell_type]['validation_sets'])
    _, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='division_balanced',
                                                              testing_exp=configuration[cell_type]['validation_sets'])
    # _, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='normal')
else:
    _, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='division', testing_exp=configuration[cell_type]['validation_sets'])
    # training_subseqs, testing_subseqs = ut.generate_training_testing_subseqs(annotation_set, mode='balanced', sampling_size=[testing_sampling_size, training_sampling_size])

testing_subseqs = random.sample(testing_subseqs, min(sampling_size, len(testing_subseqs)))
print('Number of testing samples: {}'.format(len(testing_subseqs)))
ut.sub_seqs_statistics(testing_subseqs, sub_seq_settings)
cell_ids=[]
start_rows=[]
start_cols=[]
start_frame=[]
for sub_seq in testing_subseqs:
    cell_ids.append(sub_seq['cell'])
    start_rows.append(sub_seq['first_row'])
    start_cols.append(sub_seq['first_col'])
    start_frame.append(sub_seq['start_frame'])
with open('cell_ids.txt', 'w') as f:
    for item in cell_ids:
        f.write("%s\n" % item)
with open('start_rows.txt', 'w') as f:
    for item in start_rows:
        f.write("%s\n" % item)
with open('start_cols.txt', 'w') as f:
    for item in start_cols:
        f.write("%s\n" % item)
with open('start_frame.txt', 'w') as f:
    for item in start_frame:
        f.write("%s\n" % item)
# exit()
input_dim = (sub_seq_settings['frame_size'], window_size, window_size, input_channel)

if not ut.evaluation_predictions_check(sampling_size, model_type, restore_pix):
    print('The prediction file does not exist. Generating from model...')
    if os.path.isfile('{}{}{}.h5'.format(result_path, model_type, restore_pix)):
        print('restoring from model {}{}'.format(model_type, restore_pix))
        model.load_weights('{}{}{}.h5'.format(result_path, model_type, restore_pix))

    # Parameters
    params = {'input_dim': input_dim,
              'batch_size': batch_size,
              'shuffle': False,
              'rotate': False,
              'model_note': model_note}
    # Generators
    validation_generator = MultitaskDataGenerator(testing_subseqs, annotations, output_shapes, **params)

    # model evaluation
    prediction = model.predict_generator(generator=validation_generator,
                                         use_multiprocessing=True,
                                         workers=workers,
                                         verbose=1)
    ut.save_evaluation_predictions(prediction, sampling_size, model_type, restore_pix)
else:
    print('Restoring the prediction from file ...')
    prediction = ut.read_evaluation_predictions(sampling_size, model_type, restore_pix)

df, peaks = ut.multitask_model_evaluation(prediction, annotations, testing_subseqs, input_dim, output_shapes)
