def create_fig(fig1, plot_settings):
    from skimage import img_as_float
    import os, json
    from skimage import io
    import numpy as np


    dataset_name = plot_settings['dataset_name']
    disp_frame_num = plot_settings['frame_num']

    #print('displaying frame ' + str(disp_frame_num) + ' from dataset '+dataset_name)

    with open('current_project_metadata.json') as f:
        project_metadata = json.load(f)

    datasets = project_metadata['datasets']
    this_dataset = datasets[dataset_name]

    disp_image = this_dataset['image']
    disp_vector = this_dataset['vector']
    disp_scalar = this_dataset['scalar']
    start_frame = this_dataset['starting_frame']
    end_frame = int(start_frame + this_dataset['number_frames'] -1)


    if disp_image == 'yes':
        #print('display image only')
        img_path = this_dataset['path_img']
        name_colormap = this_dataset['name_colormap']
        img_min_val = float(this_dataset['min_value_image'])
        img_max_val = float(this_dataset['max_value_image'])

        ax1f1 = fig1.add_subplot(111)
        img = img_as_float(io.imread(os.path.join(img_path, 'IMG_'+str(disp_frame_num).zfill(4)+'.tif')))
        ax1f1.imshow(img, cmap=name_colormap, vmin=img_min_val, vmax=img_max_val)
        ax1f1.set_aspect('equal')

        if disp_scalar == 'yes':
            print('display scalar on top of image')
            if disp_vector == 'yes':
                print('display vector on top of image and scalar')
        else:
            if disp_vector == 'yes':
                print('display vector on top of image')
    else:
        if disp_scalar == 'yes':
            print('display scalar only')
            if disp_vector == 'yes':
                print('display vector on top of scalar')
        else:
            if disp_vector == 'yes':
                print('display vector only')
            else:
                print('nothing to display')

    return fig1

