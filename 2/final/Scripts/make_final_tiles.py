""" This script allows to split in tiles all the training images contained in the dataset.
    To do so you have to specify the name of the dataset to be split and the one of the
    new tiled dataset to be created. The size of the tiles needs to be specified in the
    tile_size variable.
"""


from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# specify the size of the tiles you want to build
tile_size = 512

# paths
tiled_dataset = 'Final_Dataset_' + str(tile_size)
dataset_dir = 'Final_Dataset/Test_Dev'

bhi = 'Bipbip/Haricot/Images/'
bhm = 'Bipbip/Haricot/Masks/'
bmi = 'Bipbip/Mais/Images/'
bmm = 'Bipbip/Mais/Masks/'

phi = 'Pead/Haricot/Images/'
phm = 'Pead/Haricot/Masks/'
pmi = 'Pead/Mais/Images/'
pmm = 'Pead/Mais/Masks/'

rhi = 'Roseau/Haricot/Images/'
rhm = 'Roseau/Haricot/Masks/'
rmi = 'Roseau/Mais/Images/'
rmm = 'Roseau/Mais/Masks/'

whi = 'Weedelec/Haricot/Images/'
whm = 'Weedelec/Haricot/Masks/'
wmi = 'Weedelec/Mais/Images/'
wmm = 'Weedelec/Mais/Masks/'

# build tiled dataset structure
if not os.path.exists(tiled_dataset):
	os.mkdir(tiled_dataset)

def make_path(base, hi, hm, mi, mm):
	if not os.path.exists(os.path.join(tiled_dataset, base, hi)):
		os.makedirs(os.path.join(tiled_dataset, base, hi))

	if not os.path.exists(os.path.join(tiled_dataset, base, mi)):
		os.makedirs(os.path.join(tiled_dataset, base, mi))

	if base == 'Training':
		if not os.path.exists(os.path.join(tiled_dataset, base, hm)):
			os.makedirs(os.path.join(tiled_dataset, base, hm))

		if not os.path.exists(os.path.join(tiled_dataset, base, mm)):
			os.makedirs(os.path.join(tiled_dataset, base, mm))


# training
make_path('Training', bhi, bhm, bmi, bmm)
make_path('Training', phi, phm, pmi, pmm)
make_path('Training', rhi, rhm, rmi, rmm)
make_path('Training', whi, whm, wmi, wmm)

# util functions
# --------------

""" method that takes a numpy array containing the image and builds the tiles 
    of the specified size and stride
"""
def get_patches(img_arr, size=256, stride=256):

    patches_list = []
    i_max = img_arr.shape[0] // stride
    j_max = img_arr.shape[1] // stride

    for i in range(i_max):
        for j in range(j_max):
            patches_list.append(
                img_arr[
                    i * stride : i * stride + size,
                    j * stride : j * stride + size
                ]
            )

    return np.stack(patches_list)

""" method that takes a stack of tiles and plots them in the correct order
"""
def plot_patches(img_arr, org_img_size, stride=None, size=None):

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride)
    j_max = (org_img_size[1] // stride)

    print(i_max, j_max)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.005, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1

""" method that takes a stack of tiles and reconstructs the original image
    once we specify its original size and the stride and size used for
    cropping it
"""
def reconstruct_from_patches(img_arr, org_img_size, stride, size):

    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (j_max * i_max)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1

        images_list.append(img_bg)

    return np.stack(images_list)

# testing tiles
# -------------
debugging_tiles = False

if debugging_tiles:
    test_dir = 'Test_Dir'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    x = Image.open(os.path.join(dataset_dir, 'Training', rhm, 'Roseau_haricot_0008_false.png'))
    width, height = x.size

    x = x.resize(((width // tile_size)*tile_size, (height // tile_size)*tile_size), Image.NEAREST)
    
    a = np.array(x)
    x_crops = get_patches(img_arr=a, size=tile_size, stride=tile_size) # default is 256

    x_reconstructed = reconstruct_from_patches(img_arr=x_crops, org_img_size=(x.height, x.width), size=tile_size, stride=tile_size)
    #for i in range(len(x_crops)):
    #   tile = Image.fromarray(x_crops[i], 'RGB')
    #   tile.save(os.path.join(test_dir, '_TN_' + str(i) + '.png'))

    y = Image.fromarray(x_reconstructed[0], 'RGB').resize((width, height))

    plt.figure(figsize=(10,10))
    plt.imshow(y)
    plt.show()

    sys.exit()

debugging_prep = False

if debugging_prep:
	x = Image.open('Test_Dir/_TN_3.png')
	a = np.array(x)

	colours, counts = np.unique(a.reshape(-1,3), axis=0, return_counts=1)
	print(colours)

	if (a == 0).all() or ((a == 0).all() and (a == 124).all()):
		print('TOP')
		sys.exit()
	else:
		print('POT')
		sys.exit()


# tiling dataset
# --------------

""" method that saves the tiles of all the images contained in the specified
    directory. The masks are saved in png so that the initial colours are not
    modified in the saving phase.
"""
def make_tiles(base_dir, target_dir, mask):
    images_fn = next(os.walk(base_dir))[2]
    for image in images_fn:
        img = Image.open(base_dir + image)
        img = img.resize(((img.width // tile_size) * tile_size, (img.height // tile_size) * tile_size), Image.NEAREST)
        img_arr = np.array(img)

        image = os.path.splitext(image)[0]
        img_crops = get_patches(img_arr=img_arr, size=tile_size, stride=tile_size)
        for i in range(len(img_crops)):
        	tile = Image.fromarray(img_crops[i], 'RGB')

        	if mask:
        		tile.save(os.path.join(target_dir, image + '_TN_' + str(i) + '.png'))
        	else:
        		tile.save(os.path.join(target_dir, image + '_TN_' + str(i) + '.jpg'), subsampling = 0, quality=100)


# TRAINING
# --------

# bipbip
make_tiles(os.path.join(dataset_dir, bhi), os.path.join(tiled_dataset, 'Training', bhi), False)
make_tiles(os.path.join(dataset_dir, bhm), os.path.join(tiled_dataset, 'Training', bhm), True)
make_tiles(os.path.join(dataset_dir, bmi), os.path.join(tiled_dataset, 'Training', bmi), False)
make_tiles(os.path.join(dataset_dir, bmm), os.path.join(tiled_dataset, 'Training', bmm), True)

# pead
make_tiles(os.path.join(dataset_dir, phi), os.path.join(tiled_dataset, 'Training', phi), False)
make_tiles(os.path.join(dataset_dir, phm), os.path.join(tiled_dataset, 'Training', phm), True)
make_tiles(os.path.join(dataset_dir, pmi), os.path.join(tiled_dataset, 'Training', pmi), False)
make_tiles(os.path.join(dataset_dir, pmm), os.path.join(tiled_dataset, 'Training', pmm), True)

# roseau
make_tiles(os.path.join(dataset_dir, rhi), os.path.join(tiled_dataset, 'Training', rhi), False)
make_tiles(os.path.join(dataset_dir, rhm), os.path.join(tiled_dataset, 'Training', rhm), True)
make_tiles(os.path.join(dataset_dir, rmi), os.path.join(tiled_dataset, 'Training', rmi), False)
make_tiles(os.path.join(dataset_dir, rmm), os.path.join(tiled_dataset, 'Training', rmm), True)

# weedelec
make_tiles(os.path.join(dataset_dir, whi), os.path.join(tiled_dataset, 'Training', whi), False)
make_tiles(os.path.join(dataset_dir, whm), os.path.join(tiled_dataset, 'Training', whm), True)
make_tiles(os.path.join(dataset_dir, wmi), os.path.join(tiled_dataset, 'Training', wmi), False)
make_tiles(os.path.join(dataset_dir, wmm), os.path.join(tiled_dataset, 'Training', wmm), True)