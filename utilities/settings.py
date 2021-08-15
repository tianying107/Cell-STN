from lineage_classes import DataGenerator, weighted_binary_crossentropy, sequence_binary_crossentropy, adjusted_RMSE, adjusted_peak_accuracy, weighted_peak_crossentropy, weighted_accuracy_crossentropy
import os


def alloc_settings_command(commands):
    """

    :param commands: (list) The command inputs
    :return: (dict) assigned settings
    """
    settings = {}
    pix = '-c1'
    model_type = commands[1]
    settings.update({'model_type': model_type})
    if len(commands) < 2:
        raise NameError('At least one command is expected.')
    if len(commands) > 2:
        version_has_alloc_command = commands[2][0] == '-'

    if not version_has_alloc_command:
        raise NameError('The command has been deparated.')

    naming = {'gss': False, 'channel': False, 'iter': False, 'continue': False, 'speedup': False, 'cell': False,
              'blender': False, 'mask': False}
    commands = commands[2::]
    for i in range(0, len(commands), 2):
        variable = commands[i]
        value = commands[i+1]
        if not variable[0] == '-':
            raise NameError('Illegal command: "{}".'.format(variable))

        if variable == '-l':
            # setup loss function
            if value == 'bce':
                settings.update({'loss': 'binary_crossentropy'})
            elif value == 'wbc':
                settings.update({'loss': weighted_binary_crossentropy([1, 10])})
            elif value == 'wbc2':
                settings.update({'loss': weighted_binary_crossentropy([1, 2])})
            elif value == 'wbc5':
                settings.update({'loss': weighted_binary_crossentropy([1, 5])})
            elif value == 'sbe':
                settings.update({'loss': sequence_binary_crossentropy([1, 0.5])})
            elif value == 'wpc':
                settings.update({'loss': weighted_peak_crossentropy([0.1, 1])})
            elif value == 'wac':
                settings.update({'loss': weighted_accuracy_crossentropy([0.1, 1])})
            pix = '{}-{}'.format(pix, value)
        elif variable == '-outer':
            # setup outer size of the ground truth window
            settings.update({'outer_range': int(value)})
            pix = '{}-o{}'.format(pix, value)
        elif variable == '-batch':
            # setup batch size
            settings.update({'batch_size': int(value)})
        elif variable == '-continue':
            # setup restored weights
            settings.update({'train_mode': 'continue', 'restore_pix': value})
            naming['continue'] = True
        elif variable == '-epoch':
            settings.update({'epochs': int(value)})
        elif variable == '-c':
            # setup restored weights
            settings.update({'input_channel': int(value)})
            naming['channel'] = True
        elif variable == '-gss':
            # enable data generator with GSS
            naming['gss'] = value.lower() in ['true', '1', 't', 'y', 'yes']
            settings.update({'gss': value.lower() in ['true', '1', 't', 'y', 'yes']})
        elif variable == '-iter':
            # given iteration note
            naming['iter'] = value
        elif variable == '-su':
            # enable data augmentation: speedup
            settings.update({'speedup': value.lower() in ['true', '1', 't', 'y', 'yes']})
            naming['speedup'] = value
        elif variable == '-cell':
            # enable data augmentation: speedup
            settings.update({'cell_type': value})
            naming['cell'] = value
        elif variable == '-blender':
            # enable data mix up
            settings.update({'blender': value.lower() in ['true', '1', 't', 'y', 'yes']})
            naming['blender'] = value.lower() in ['true', '1', 't', 'y', 'yes']
        elif variable == '-mask':
            # enable mask ground truth
            settings.update({'mask': value.lower() in ['true', '1', 't', 'y', 'yes']})
            naming['mask'] = value.lower() in ['true', '1', 't', 'y', 'yes']

    # naming in order
    if naming['channel']:
        pix = '-c{}'.format(settings['input_channel'])
    if naming['iter']:
        pix = '{}-i{}'.format(pix, naming['iter'])  # i.e. -c3-i2
    if naming['gss']:
        pix = '{}-gss'.format(pix)
    if naming['speedup']:
        pix = '{}-su'.format(pix)
    if naming['cell']:
        pix = '{}-{}'.format(pix, naming['cell'])
    if naming['blender']:
        pix = '{}-bl'.format(pix)
    if naming['mask']:
        pix = '{}-mask'.format(pix)
    if naming['continue']:
        weight_counter = len([benchmark for benchmark in os.listdir(os.path.expanduser('./benchmarks/')) if
                              model_type + pix in benchmark]) + 1
        pix = '{}-con-{}'.format(pix, weight_counter)
    settings.update({'pix': pix})
    return settings


def get_settings(cell_type='HCT116', input_channel=1, epochs=5, batch_size=2, train_mode='start', restore_pix=None, pix='-c1',
                 loss='binary_crossentropy', outer_range=None, speedup=False, model_type=None, gss=False, blender=False, mask=None):
    gt_paras = {}
    if outer_range is not None:
        gt_paras.update({'outer_range': outer_range})
    if mask is not None:
        gt_paras.update({'mask_ground_truth': mask})
    if train_mode == 'start':
        weight_counter = len([benchmark for benchmark in os.listdir(os.path.expanduser('./benchmarks/')) if
                              model_type + pix in benchmark]) + 1
        pix = '{}-{}'.format(pix, weight_counter)
    return cell_type, input_channel, epochs, batch_size, pix, train_mode, restore_pix, loss, gt_paras, speedup, gss, blender
