import numpy as np
import os
import utility as ut
import pandas as pd
from utilities import ground_truth_convert
from scipy import stats

def get_evaluation_prediction_name(sampling_size, model_type, restore_pix):
    sampling_pix = '-s{}'.format(sampling_size)
    configuration = ut.read_configuration()
    evaluation_path = configuration['evaluation_path']
    return '{}p-{}{}{}.npy'.format(evaluation_path, model_type, restore_pix, sampling_pix)


def generate_hard_evaluation_predictions(annotations, testing_subseqs, input_dim, output_shape, additionals={'inner_range': 0, 'outer_range': 0}):
    predictions = np.zeros((len(testing_subseqs), *output_shape[1::]))
    for batch, sub_seq in enumerate(testing_subseqs):
        ground_truth = ut.select_annotation_for_subseq(annotations, sub_seq)
        start_f = sub_seq['start_frame']
        true_col = ground_truth['cols'][start_f]
        true_row = ground_truth['rows'][start_f]

        coordinate_correct = ground_truth_convert.ground_truth_scaling(true_row, true_col, sub_seq['offset_x'],
                                                                       sub_seq['offset_y'], input_dim,
                                                                       (None, *input_dim))
        hard_prediction_map = ground_truth_convert.coordinates_to_confidence_map(coordinate_correct, output_shape,
                                                           type='square', **additionals)

        hard_prediction = np.expand_dims(np.repeat(hard_prediction_map[np.newaxis, ], output_shape[1], axis=0), axis=-1)

        predictions[batch, ] = hard_prediction

    return predictions


def evaluation_predictions_check(sampling_size, model_type, restore_pix):
    file_name = get_evaluation_prediction_name(sampling_size, model_type, restore_pix)
    return os.path.isfile(file_name)


def save_evaluation_predictions(prediction, sampling_size, model_type, restore_pix):
    configuration = ut.read_configuration()
    evaluation_path = configuration['evaluation_path']
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    file_name = get_evaluation_prediction_name(sampling_size, model_type, restore_pix)
    np.save(file_name, prediction)
    print('The prediction file is saved as {}.'.format(file_name))


def read_evaluation_predictions(sampling_size, model_type, restore_pix):
    file_name = get_evaluation_prediction_name(sampling_size, model_type, restore_pix)
    return np.load(file_name)


def event_level_evaluation(y_pred, y_true):
    value_true_positive_threshold = 0.4
    pred_value = np.max(np.max(np.max(y_pred, axis=1), axis=1), axis=0)
    pred_value = np.greater(pred_value, value_true_positive_threshold).astype('float32')
    gt_value = np.max(np.max(np.max(y_true, axis=1), axis=1), axis=0)
    gt_value = np.greater(gt_value, value_true_positive_threshold).astype('float32')

    true_positive = np.equal(pred_value * gt_value, 1).astype('float32')
    true_negative = np.equal(pred_value + gt_value, 0).astype('float32')
    false_positve = np.equal(pred_value - gt_value, 1).astype('float32')
    false_negative = np.equal(pred_value - gt_value, -1).astype('float32')
    # print(true_positive)
    # print(true_negative)
    # print(false_positve)
    # print(false_negative)
    return np.sum(true_positive), np.sum(true_negative), np.sum(false_positve), np.sum(false_negative)




def multitask_model_evaluation(predict, annotations, testing_subseqs, input_dim, output_shapes, temporal_threshold=2,
                               spatial_threshold=20, blender=False):
    blender = True
    # currently, the function only support single output.
    predict = np.expand_dims(predict[:, :, :, :, 0], axis=-1)
    predict = np.squeeze(predict)

    peaks_in_frame = np.max(np.max(predict, axis=2), axis=2)
    pred_cols = np.argmax(np.max(predict, axis=2), axis=2)
    pred_rows = np.argmax(np.max(predict, axis=3), axis=2)
    pred_frames = np.argmax(peaks_in_frame, axis=1)

    # print(pred_frames)
    in_scope = np.ones(pred_frames.shape)
    out_scope = np.zeros(pred_frames.shape)
    gt_frames = np.zeros(pred_frames.shape)
    visual_no_change = np.zeros(pred_frames.shape)

    spatial_distances = np.zeros(pred_frames.shape)
    pred_values = np.zeros(pred_frames.shape)
    gt_values = np.zeros(pred_values.shape)

    v_TP = np.zeros(pred_values.shape)
    v_TN = np.zeros(pred_values.shape)
    v_FP = np.zeros(pred_values.shape)
    v_FN = np.zeros(pred_values.shape)
    for batch in range(predict.shape[0]):
        sub_seq = testing_subseqs[batch]
        if 'death' in output_shapes:
            ut.death_sequence_valid_check(sub_seq, input_dim, annotations)
        if 'division' in output_shapes:
            # find related sub seq
            _, related_sub_seq = ground_truth_convert.get_related_sub_seq(annotations, sub_seq, output_shapes['division'])
            visual_no_change[batch] = related_sub_seq is None

        ground_truth = ut.select_annotation_for_subseq(annotations, sub_seq)
        if 'division' in output_shapes:
            gt_division = ut.generate_division_truth_sequence(ground_truth, sub_seq, input_dim,
                                                              output_shapes['division'],
                                                              spatial=(len(output_shapes['division']) > 2))
        elif 'death' in output_shapes:
            gt_division = ut.generate_death_truth_sequence(ground_truth, sub_seq, input_dim,
                                                              output_shapes['death'],
                                                              spatial=(len(output_shapes['death']) > 2))
        gt_division = np.expand_dims(gt_division[:, :, :, 0], axis=-1)

        pred_values[batch] = peaks_in_frame[batch, pred_frames[batch]]

        if blender:
            TP, TN, FP, FN = event_level_evaluation(predict[batch, ], gt_division)
            v_TP[batch] = TP
            v_TN[batch] = TN
            v_FP[batch] = FP
            v_FN[batch] = FN
            # if FN == 1:
            #     print(sub_seq)
            #     print(pred_values[batch])

        if np.max(np.squeeze(np.max(np.max(gt_division, axis=1), axis=1))) != 1:
            in_scope[batch] = 0
            if np.max(np.squeeze(np.max(np.max(gt_division, axis=1), axis=1))) == 0:
                # print('ground truth out of scope completely!')
                out_scope[batch] = 1
        gt_values[batch] = np.max(np.squeeze(np.max(np.max(gt_division, axis=1), axis=1)))

        gt_frame = np.argmax(np.squeeze(np.max(np.max(gt_division, axis=1), axis=1)), axis=0)
        gt_frames[batch] = gt_frame

        gt_row = np.squeeze(np.argmax(np.max(gt_division[gt_frame, :, :, 0], axis=1), axis=0))
        gt_col = np.squeeze(np.argmax(np.max(gt_division[gt_frame, :, :, 0], axis=0), axis=0))
        spatial_distances[batch] = euclid_distance((gt_row, gt_col), (pred_rows[batch, pred_frames[batch]], pred_cols[batch, pred_frames[batch]]))


    # print(gt_frames)

    temporal_distances = np.abs((gt_frames - pred_frames))

    temporal_on_track = temporal_distances<=temporal_threshold
    spatial_on_track = spatial_distances<=spatial_threshold
    on_track = np.array(temporal_on_track * spatial_on_track, dtype='float32')


    adjusted_on_track = on_track[(in_scope == 1) & (visual_no_change == 0)]
    adjusted_temporal_distances = temporal_distances[(in_scope == 1) & (visual_no_change == 0)]
    adjusted_spatial_distances = spatial_distances[(in_scope == 1) & (visual_no_change == 0)]

    print('spatial distance: {}'.format(np.mean(spatial_distances[(in_scope == 1)])))
    print('temporal distance: {}'.format(np.mean(temporal_distances[(in_scope == 1)])))
    print('peak accuracy: {}'.format(np.mean(on_track[(in_scope == 1)])))

    if blender:
        # v_TP = v_TP[(visual_no_change == 0) | (v_TN == 1) | (v_FP == 1)]
        # v_FN = v_FN[(visual_no_change == 0) | (v_TN == 1) | (v_FP == 1)]
        # on_track = on_track[(visual_no_change == 0) | (v_TN == 1) | (v_FP == 1)]
        print('Total samples: {}'.format(len(v_TP)))
        print('Event-level True Positives: {}'.format(np.sum(v_TP)))
        print('True Negative: {}'.format(np.sum(v_TN)))
        print('False Positives: {}'.format(np.sum(v_FP)))
        print('False Negative: {}'.format(np.sum(v_FN)))
        print('Total True Postives: {}'.format(np.sum(on_track*v_TP)))
        precision = np.sum(on_track*v_TP) / (np.sum(on_track*v_TP) + np.sum(v_FP))
        recall = np.sum(on_track*v_TP) / (np.sum(on_track*v_TP) + np.sum(v_FN))
        f_score = 2*(precision*recall)/(precision+recall)
        specificity = np.sum(v_TN) / (np.sum(v_TN) + np.sum(v_FP))
        acc = (np.sum(on_track*v_TP) + np.sum(v_TN))/len(v_TP)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('f1 score: {}'.format(f_score))
        print('specificity: {}'.format(specificity))
        print('accuracy: {}'.format(acc))
        exit()

    print(pred_values)
    print(out_scope)
    print(visual_no_change)
    # print(on_track)
    # print(adjusted_on_track)
    adjusted_peak_accuracy = np.mean(adjusted_on_track)
    print('total available samples: {}'.format(adjusted_on_track.shape[0]))
    print('total out of scope:{}'.format(np.sum(1-in_scope)))
    print('total non-visible division:{}'.format(np.sum(visual_no_change)))

    print('adjusted spatial distance: {}'.format(np.mean(adjusted_spatial_distances)))
    print('adjusted temporal distance: {}'.format(np.mean(adjusted_temporal_distances)))
    print('adjusted peak accuracy: {}'.format(adjusted_peak_accuracy))

    return temporal_distances, peaks_in_frame


def model_evaluation(predict, annotations, testing_subseqs, input_dim, output_shape):
    rows = annotations['rows']
    cols = annotations['cols']

    total_travel_distance = ground_truth_travel_distance(testing_subseqs)

    predict = np.squeeze(predict)
    # print(predict.shape)

    maxs = np.amax(np.amax(predict, axis=2), axis=2)
    b = np.arange(predict.shape[0])
    f = np.arange(predict.shape[1])
    fv, bv = np.meshgrid(f, b)
    bv = np.squeeze(np.reshape(bv, (1, -1)))
    fv = np.squeeze(np.reshape(fv, (1, -1)))
    distance = np.zeros((predict.shape[0], predict.shape[1]))
    in_scope = np.ones(distance.shape)
    single_travel_distance = np.ones(distance.shape)
    peaks = np.zeros((predict.shape[0], predict.shape[1], 2))
    first_fail = np.zeros((predict.shape[0], predict.shape[1]))
    for ba, fa in zip(bv, fv):
        peak = np.where(predict[ba, fa,] == maxs[ba, fa,])
        peak = list((peak[0][0], peak[1][0]))
        peaks[ba, fa, ] = np.array([peak[0], peak[1]])

        sub_seq = testing_subseqs[ba]
        # print(sub_seq)
        cell = sub_seq['cell']
        start = sub_seq['start_frame']
        row = rows[cell, start + fa]
        col = cols[cell, start + fa]
        coordinate_correct = ground_truth_convert.ground_truth_scaling(row, col,
                                                                       sub_seq['offset_x'], sub_seq['offset_y'],
                                                                       input_dim, output_shape)
        if fa == 0:
            single_travel_distance[ba, fa] = 0
        else:
            previous_row = rows[cell, start + fa - 1]
            previous_col = cols[cell, start + fa - 1]
            previous_coordinate_correct = ground_truth_convert.ground_truth_scaling(previous_row, previous_col,
                                                                           sub_seq['offset_x'], sub_seq['offset_y'],
                                                                           input_dim, output_shape)
            single_travel_distance[ba, fa] = euclid_distance(coordinate_correct, previous_coordinate_correct)
        if ground_truth_convert.out_of_scope_check(coordinate_correct, output_shape):
            in_scope[ba, fa] = 0

        distance[ba, fa] = np.sum((np.array(coordinate_correct) - np.array(peak)) ** 2)

        if fa>0:
            if np.sqrt(distance[ba, fa]) >10:
                # when the current frame failed
                first_fail[ba, fa] = np.sqrt(distance[ba, fa-1]) <= 10

    distance = np.sqrt(distance)
    RMSE = np.mean(distance)
    on_track = np.where(distance <= 10, np.ones(distance.shape), 0)
    accuracy_frame = np.mean(on_track)
    accuracy_track = np.mean(np.where(np.mean(on_track, axis=1) == 1, np.ones(np.mean(on_track, axis=1).shape), 0))
    accuracy_end = np.mean(on_track[:, -1])
    # print(on_track)
    print('frame-wise accuracy: {}'.format(accuracy_frame))
    print('track-wise accuracy: {}'.format(accuracy_track))
    print('end-point accuracy: {}'.format(accuracy_end))
    print('RMSE: {}'.format(RMSE))

    max_value = np.max(np.max(predict, axis=2), axis=2)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame({'distance': np.squeeze(np.reshape(distance, (1, -1)))})
    df['travel'] = np.squeeze(np.reshape(single_travel_distance, (1, -1)))
    df['on_track'] = np.squeeze(np.reshape(on_track, (1, -1)))
    df['in_scope'] = np.squeeze(np.reshape(in_scope, (1, -1)))
    df['max_value'] = np.squeeze(np.reshape(max_value, (1, -1)))
    df['first_fail'] = np.squeeze(np.reshape(first_fail, (1, -1)))
    df['frame'] = np.squeeze(np.reshape(fv, (1, -1)))


    with open('in_scope.txt', 'w') as f:
        for row in in_scope:
            for item in row:
                f.write("%s," % item)
            f.write("\n")
    # adjusted on track which counts out of scope as true positive
    distance = in_scope * distance
    RMSE = np.mean(distance)
    on_track = np.where(distance <= 10, np.ones(distance.shape), 0)

    adj_accuracy_frame = np.mean(on_track)
    accuracy_track = np.mean(np.where(np.mean(on_track, axis=1) == 1, np.ones(np.mean(on_track, axis=1).shape), 0))
    accuracy_end = np.mean(on_track[:, -1])
    adj_accuracy_frame = accuracy_frame/(1 - (adj_accuracy_frame-accuracy_frame))
    print('adjusted frame-wise accuracy: {}'.format(adj_accuracy_frame))
    print('adjusted track-wise accuracy: {}'.format(accuracy_track))
    print('adjusted end-point accuracy: {}'.format(accuracy_end))
    print('adjusted RMSE: {}'.format(RMSE))

    return df, peaks

def ground_truth_travel_distance(testing_subseqs):
    first_points = np.zeros((len(testing_subseqs), 2))
    last_points = np.zeros((len(testing_subseqs), 2))
    for index, sub_seq in enumerate(testing_subseqs):
        # print(sub_seq)
        first_col = sub_seq['first_col']
        first_row = sub_seq['first_row']
        last_col = sub_seq['last_col']
        last_row = sub_seq['last_row']
        first_points[index, ] = np.array((first_row, first_col))
        last_points[index,] = np.array((last_row, last_col))
    distance = np.sqrt(np.sum((first_points - last_points) ** 2, axis=1))
    # print(distance)
    return distance


def euclid_distance(point_a, point_b):
    return np.sqrt(np.sum((np.array(point_a) - np.array(point_b)) ** 2))


def evaluation_results_check(model_type, restore_pix):
    file_name = get_evaluation_results_name(model_type, restore_pix)
    return os.path.isfile(file_name)


def get_evaluation_results_name(model_type, restore_pix):
    configuration = ut.read_configuration()
    evaluation_path = configuration['evaluation_path']
    return '{}p-{}{}.pkl'.format(evaluation_path, model_type, restore_pix)


def save_evaluation_results(df, model_type, restore_pix):
    configuration = ut.read_configuration()
    evaluation_path = configuration['evaluation_path']
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    file_name = get_evaluation_results_name(model_type, restore_pix)
    df.to_pickle(file_name)
    print('The evaluation result data frame is saved as {}.'.format(file_name))


def read_evaluation_results(model_type, restore_pix):
    file_name = get_evaluation_results_name(model_type, restore_pix)
    return pd.read_pickle(file_name)


def result_analysis(df):
    df = df[(df.in_scope == 1) & ((df.on_track == 1)|(df.first_fail == 1))]
    print(df)
    plot_violin(df)


def plot_violin(df, compare_feature):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax2 = plt.subplots()

    print(df.groupby('on_track').mean())
    print(df.groupby('on_track').std())
    print(df.groupby('on_track').count())
    sns.violinplot('on_track', compare_feature, data=df, ax=ax2)
    ax2.set_title('{} comparison'.format(compare_feature))

    plt.show()


def t_test(df, compare_feature):
    t, p = stats.ttest_ind(df[df['on_track'] == 1][compare_feature], df[df['on_track'] == 0][compare_feature])
    return t, p

