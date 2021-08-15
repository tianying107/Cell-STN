from PIL import Image
import numpy as np
import os
import utility

def tif2npy(cell_type, configuration):
    '''

    :param configuration: indicates the path of raw images
    :return:
    '''
    raw_image_path = '{}{}/'.format(configuration['raw_image_path'], cell_type)
    temp_image_path = '{}{}/'.format(configuration['temp_image_path'], cell_type)
    channel_dict = configuration[cell_type]['channels_name']
    channels = len(channel_dict)

    # iterate experiments
    for experiment in configuration[cell_type]['experiments']:
        print('Converting images from experiment {}'.format(experiment['id']))
        exp_id = experiment['id']

        im = Image.open('{}Pos{}{}.TIF'.format(raw_image_path, exp_id, channel_dict[0]))
        h,w = np.shape(im)

        # if utility.deploy_check(configuration):
        #     imarray = np.zeros((im.n_frames, h, w, channels))
        # else:
        imarray = np.zeros((experiment['frames'], h, w, channels))

        for c in range(channels):
            im = Image.open('{}Pos{}{}.TIF'.format(raw_image_path, exp_id, channel_dict[c]))
            frames = im.n_frames

            # develop mode set the total frames to 5
            if not utility.deploy_check(configuration):
                frames = experiment['frames']

            for i in range(frames):
                im.seek(i)
                imarray[i, :, :, c] = np.array(im)
            min_value = np.min(imarray[:, :, :, c])
            max_value = np.max(imarray[:, :, :, c])
            imarray[:, :, :, c] = (imarray[:, :, :, c] - min_value) / (max_value - min_value)

        # the np_array for the entire experiment
        imarray = imarray.astype(float)

        # store the array as npy file, first check the directory
        if not os.path.isdir(configuration['temp_image_path']):
            try:
                os.mkdir(configuration['temp_image_path'])
            except OSError:
                print("Creation of the directory {} failed".format(configuration['temp_image_path']))
            else:
                print("Successfully created the directory {} ".format(configuration['temp_image_path']))

        if not os.path.isdir(temp_image_path):
            try:
                os.mkdir(temp_image_path)
            except OSError:
                print("Creation of the directory {} failed".format(temp_image_path))
            else:
                print("Successfully created the directory {} ".format(temp_image_path))

        file_name = 'Pos{}_c{}f{}'.format(experiment['id'], channels, frames)
        np.save('{}{}'.format(temp_image_path, file_name), imarray)
        print('Successfully created the file {}.'.format(file_name))