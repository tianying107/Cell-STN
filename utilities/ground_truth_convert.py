import utility
import pandas as pd
import numpy as np
import random
from PIL import Image


def coordinates_to_confidence_map(coordicates, output_shape, type='square', inner_range=1, outer_range=5):
    """
    This function generate confidence map from coordinates. The default type is square.
    :param coordicates: (list) [row, col]
    :param output_shape: (list) (batch, frames, rows, cols, channels)
    :param type: (string) 'square' or 'gaussian'
    :param inner_range: (int) Optional. This is only required when type is 'square', which indicates the inner size of
    the square window
    :param outer_range: (int) Optional. This is only required when type is 'square', which indicates the outer size of
    the square window
    :return: confidence_map: (np_array)
    """
    # coordicates: [row, col]
    # output_shape: [batch, frames, rows, cols, channels]
    # print(outer_range)
    # exit()
    gt_row = coordicates[0]
    gt_col = coordicates[1]
    total_row = output_shape[2]
    total_col = output_shape[3]
    if type == 'gaussian':
        confidence_map = utility.coordinate2map.xy2gaussian(gt_col, gt_row, total_col, total_row)
    elif type == 'square':
        confidence_map = utility.coordinate2map.xy2square(gt_col, gt_row, total_col, total_row, inner_range=inner_range,
                                                          outer_range=outer_range)
    elif type == 'square_single':
        confidence_map = utility.coordinate2map.xy2square(gt_col, gt_row, total_col, total_row, inner_range=1,
                                                          outer_range=1)
    return confidence_map


def select_mask(sub_seq, coordicates, current_frame):
    configuration = utility.read_configuration()
    mask_path = configuration[sub_seq['cell_type']]['mask_path']

    start = sub_seq['start_frame']
    mask_name = '{}{}/man_track{:03d}.tif'.format(mask_path, sub_seq['experiment'], start + current_frame)
    im = np.array(Image.open(mask_name))
    mask = np.array(im == im[coordicates[0], coordicates[1]], dtype='float')
    return mask[sub_seq['offset_y']:sub_seq['offset_y'] + 256,
                           sub_seq['offset_x']:sub_seq['offset_x'] + 256]

def ground_truth_scaling(gt_row, gt_col, offset_x, offset_y, input_dim, output_shape):
    """
    This function project the ground truth coordinates at the original scale to the scale of model output.
    :param gt_row:
    :param gt_col:
    :param offset_x:
    :param offset_y:
    :param input_dim:
    :param output_shape:
    :return:
    """
    # input_dim: [frames, rows, cols, channels]
    # output_shape: [batch, frames, rows, cols, channels]

    gt_row -= offset_y+1
    gt_col -= offset_x+1

    row_ratio = output_shape[2]/input_dim[1]
    col_ratio = output_shape[3]/input_dim[2]

    gt_row *= row_ratio
    gt_col *= col_ratio

    return int(round(gt_row)), int(round(gt_col))


def out_of_scope_check(correct_coordinate, output_shape):
    # gt_row and gt_col are coordinates after scaling
    # input_dim: [frames, rows, cols, channels]
    # output_shape: [batch, frames, rows, cols, channels]
    gt_row = correct_coordinate[0]
    gt_col = correct_coordinate[1]
    if (gt_row < 0) | (gt_col < 0) | (gt_row >= output_shape[2]) | (gt_col >= output_shape[3]):
        return True
    return False


def invalid_scope_check(correct_coordinate, correct_frame, input_dim, sub_seq, annotations):
    # gt_row and gt_col are coordinates after scaling
    # input_dim: [frames, rows, cols, channels]
    # output_shape: [batch, frames, rows, cols, channels]
    # random.seed(42)
    total_rows = 1024
    total_cols = 1024
    window_size = 256
    spatial_margin = 15
    temporal_margin = 1
    gt_row = correct_coordinate[0]
    gt_col = correct_coordinate[1]
    if (gt_row < spatial_margin) | (gt_col < spatial_margin) | (gt_row >= input_dim[1] - spatial_margin) | (
            gt_col >= input_dim[2] - spatial_margin) | (correct_frame <= temporal_margin):
        new_offset_row = min(total_rows - window_size, max(0, sub_seq['offset_y'] - (max(0, spatial_margin - gt_row) + min(0, input_dim[1] - spatial_margin - gt_row))))
        new_offset_col = min(total_cols - window_size, max(0, sub_seq['offset_x'] - (max(0, spatial_margin - gt_col) + min(0, input_dim[2] - spatial_margin - gt_col))))
        sub_seq['offset_y'] = new_offset_row
        sub_seq['offset_x'] = new_offset_col
        if correct_frame <= temporal_margin:
            new_start_frame = max(0, sub_seq['start_frame'] - random.randint(temporal_margin, temporal_margin+2))
            sub_seq['start_frame'] = new_start_frame
            sub_seq['first_row'] = annotations['rows'][sub_seq['cell'], new_start_frame]
            sub_seq['first_col'] = annotations['cols'][sub_seq['cell'], new_start_frame]
    return sub_seq


def contain_division_check(sub_seq, ground_truth, frames=10, speed_factor=1):
    start_f = sub_seq['start_frame']
    adjusted_division = ground_truth['division'][start_f:start_f + speed_factor*frames]
    return max(adjusted_division) == 1


def contain_death_check(sub_seq, ground_truth, frames=10, speed_factor=1):
    start_f = sub_seq['start_frame']
    adjusted_death = ground_truth['death'][start_f:start_f + speed_factor*frames]
    return max(adjusted_death) == 1


def select_normal_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    df = df[(df.include_division == 0) & (df.include_death == 0) & (df.first_col != 0) & (df.first_row != 0)]
    return df.to_dict(orient='record')


def count_normal_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    df = df[(df.include_division == 0) & (df.include_death == 0) & (df.first_col != 0) & (df.first_row != 0)]
    return len(df.to_dict(orient='record'))


def select_division_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    # df = df[(df.include_division == 1)]
    df = df[(df.include_true_division == 1)]
    return df.to_dict(orient='record')


def count_division_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    df = df[(df.include_division == 1)]
    return len(df.to_dict(orient='record'))


def select_death_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    df = df[(df.include_death == 1)]
    return df.to_dict(orient='record')


def count_death_subsequence(sub_seqs):
    df = pd.DataFrame(sub_seqs)
    df = df[(df.include_death == 1)]
    return len(df.to_dict(orient='record'))


def select_config_by_exp(configuration, cell_type, exp_id):
    df = pd.DataFrame(configuration[cell_type]['experiments'])
    df = df[df.id.astype('str') == '{}'.format(exp_id)]
    return df.to_dict(orient='record')[0]


def split_subseq_by_exp(sub_seqs, exp_id):
    if len(sub_seqs) > 0:
        df = pd.DataFrame(sub_seqs)
        df_y = df[df.experiment.isin(exp_id)]
        df_n = df[~df.experiment.isin(exp_id)]
        return df_y.to_dict(orient='record'), df_n.to_dict(orient='record')
    else:
        return [], []


def correct_division_annotations(annotations, cell_type):
    division = annotations['division']
    true_division = division.copy()
    if cell_type == 'MCF7':
        division_point = np.where(division == 1)
        for cell, frame in zip(division_point[0], division_point[1]):
            if frame + 10 > len(division):
                true_division[cell, frame] = 0
            elif np.sum(division[cell, frame:frame + 10]) != 1:
                true_division[cell, frame] = 0
            else:
                true_division[cell, frame] = 0
                true_division[cell, frame-1] = 1
    new_annotations = annotations.copy()
    new_annotations['division'] = true_division
    return new_annotations


def correct_death_annotations(annotations, cell_type):
    death = annotations['death']
    true_death = death.copy()
    if cell_type == 'HCT116':
        death_point = np.where(death == 1)
        for cell, frame in zip(death_point[0], death_point[1]):
            true_death[cell, frame] = 0
            true_death[cell, frame+1] = 1
    new_annotations = annotations.copy()
    new_annotations['death'] = true_death
    return new_annotations


def segment_cell_subsequence(annotations, experiments, offset_x, offset_y, window_size, frame_size, frame_step):
    # need to convert the exp data in config to exp_list
    df_config = pd.DataFrame(experiments)
    exp_list = list(df_config['id'])

    rows = annotations['rows']
    cols = annotations['cols']
    death = annotations['death']
    division = annotations['division']
    cell_type = annotations['cell_type']

    true_division = correct_division_annotations(annotations, cell_type)['division']

    # output_shape: (list) (batch, frames, rows, cols, channels)
    output_shape = (None, frame_size, window_size, window_size, 1)
    # input_dim (frames, rows, cols, channels)
    input_dim = (frame_size, window_size, window_size, 1)
    sub_seqs = []

    for cell in range(annotations['experiment'].shape[0]):
        # check exp id is valid
        exp_id = '{:02d}'.format(annotations['experiment'][cell, 0])
        if exp_id not in exp_list:
            continue
        # iterate per cell, the cell indicates the row of data in each matrix.
        # Using offset x, y, f, window_size, frame_size to locate a image sub_sequence (f:f+10, y:y+256, x:x+256)
        # first, check the first frame whether contain target cell by comparing the spatial boundary
        for start_frame in range(0, rows.shape[1], frame_step):
            first_coord = (rows[cell, start_frame], cols[cell, start_frame])

            ground_truth = ground_truth_scaling(first_coord[0], first_coord[1], offset_x, offset_y,
                                                                     input_dim,
                                                                     output_shape)
            # check out of scope
            if out_of_scope_check(ground_truth, output_shape):
                continue

            # check already dead
            if (first_coord[0] == 0) & (first_coord[1] == 0):
                continue

            # check last frame is smaller than the total frame
            if start_frame + frame_size > int(df_config[df_config.id == exp_id].frames):
                continue
            last_coord = (rows[cell, start_frame + frame_size - 1], cols[cell, start_frame + frame_size - 1])

            sub_seq = {'offset_x': offset_x, 'offset_y': offset_y, 'start_frame': start_frame,
                       'first_col': first_coord[1], 'first_row': first_coord[0], 'last_col': last_coord[1],
                       'last_row': last_coord[0], 'cell': cell,
                       'experiment': '{:02d}'.format(annotations['experiment'][cell, 0]), 'target_at_begin': 1,
                       'include_division': 0, 'include_true_division': 0, 'include_death': 0, 'cell_type': cell_type}

            # then, check the entire sub_sequence whether contain division and death event by comparing the temporal boundary
            if 1 in division[cell, start_frame: start_frame + frame_size]:
                sub_seq['include_division'] = 1
            if 1 in true_division[cell, start_frame: start_frame + frame_size - 1]:
                sub_seq['include_true_division'] = 1
            if 1 in death[cell, start_frame: start_frame + frame_size]:
                sub_seq['include_death'] = 1

            sub_seqs.append(sub_seq)
    # print(len(sub_seqs))
    # print(sub_seqs)
    if len(sub_seqs) > 0:
        df = pd.DataFrame(sub_seqs)
        # reduce duplicate caused by the double counting of the mother cell
        df = df.drop_duplicates(subset=['first_col', 'first_row', 'last_col', 'last_row', 'start_frame', 'experiment'])
        # {'offset_x': 256, 'offset_y': 256, 'start_frame': 125, 'first_col': 338, 'first_row': 394, 'cell': 2, 'experiment': 76, 'target_at_begin': 1, 'include_division': 0, 'include_death': 0}
        return df.to_dict(orient='record')
    else:
        return sub_seqs


def get_training_name(sub_seq_settings):
    window_size = sub_seq_settings['window_size']
    window_step = sub_seq_settings['window_step']
    frame_size = sub_seq_settings['frame_size']
    frame_step = sub_seq_settings['frame_step']
    return 'anno_f{}_{}_w{}_{}'.format(frame_size, frame_step, window_size, window_step)


def get_related_sub_seq(annotations, sub_seq, output_shape):
    """
    This function return a sub_seq which related to the given sub_seq. This related sub_seq is used to find the
    children cells after the target cell divided.
    :param annotations:
    :param sub_seq:
    :param output_shape:
    :return:
    """
    if annotations is not None:
        start = sub_seq['start_frame']
        frames = output_shape[1]

        # find out the related seq
        start_row = annotations['rows'][:, start]
        start_col = annotations['cols'][:, start]
        end_row = annotations['rows'][:, start + frames - 1]
        end_col = annotations['cols'][:, start + frames - 1]

        # ====below code is used for extend the last frame====
        # end_row = annotations['rows'][:, start + frames]
        # end_col = annotations['cols'][:, start + frames]
        # new_last_row = annotations['rows'][sub_seq['cell'], start + frames]
        # new_last_col = annotations['cols'][sub_seq['cell'], start + frames]
        # ====end====


        exp_id = sub_seq['experiment']
        sub_seqs = {'first_row': start_row, 'first_col': start_col, 'last_row': end_row, 'last_col': end_col,
                    'experiment': annotations['experiment'][:, 0],
                    'cell': np.arange(annotations['experiment'].shape[0])}
        df = pd.DataFrame(sub_seqs)
        df = df[(df.first_row == sub_seq['first_row']) & (df.first_col == sub_seq['first_col']) & (
                (df.last_row != sub_seq['last_row']) | (df.last_col != sub_seq['last_col'])) & (
                        df.experiment.astype('str') == exp_id)]
        # ====below code is used for extend the last frame====
        # df = df[(df.first_row == sub_seq['first_row']) & (df.first_col == sub_seq['first_col']) & (
        #         (df.last_row != new_last_row) | (df.last_col != new_last_col)) & (
        #                 df.experiment == exp_id)]
        # ====end====

        df = df.drop_duplicates(subset=['first_col', 'first_row', 'last_col', 'last_row', 'experiment'])
        related_sub_seq = df.to_dict(orient='record')
        if len(related_sub_seq) > 0:
            related_sub_seq = related_sub_seq[0]
            related_sub_seq.update(
                {'offset_x': sub_seq['offset_x'], 'offset_y': sub_seq['offset_y'],
                 'start_frame': sub_seq['start_frame']})
            # find related ground truth
            related_ground_truth = utility.select_annotation_for_subseq(annotations, related_sub_seq)

            return related_ground_truth, related_sub_seq
    return None, None

