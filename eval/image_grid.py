import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys

def square_grid(array, grid_cols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//grid_cols
    assert nindex == nrows*grid_cols
    result = (array.reshape(nrows, grid_cols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*grid_cols, intensity))
    return result

def save_image_grids(model_results_image_dir, grid_dim=2, offset=0):
    realA_list = []
    fakeA_list = []
    realB_list = []
    fakeB_list = []

    for i in range(grid_dim ** 2):
        image_index = 2 * (i + offset)
        realA_list.append(Image.open(os.path.join(model_results_image_dir, "real" + str(image_index) + ".jpg")))
        fakeA_list.append(Image.open(os.path.join(model_results_image_dir, "fake" + str(image_index) + ".jpg")))

        image_index = 2 * (i + offset) + 1
        realB_list.append(Image.open(os.path.join(model_results_image_dir, "real" + str(image_index) + ".jpg")))
        fakeB_list.append(Image.open(os.path.join(model_results_image_dir, "fake" + str(image_index) + ".jpg")))

    fig = plt.figure()
    plt.axis('off')

    ax1 = fig.add_subplot(2, 2, 1)
    realA_grid = square_grid(np.stack(realA_list), grid_dim)
    ax1.imshow(realA_grid)
    ax2 = fig.add_subplot(2, 2, 2)
    realB_grid = square_grid(np.stack(realB_list), grid_dim)
    ax2.imshow(realB_grid)
    ax3 = fig.add_subplot(2, 2, 3)
    fakeA_grid = square_grid(np.stack(fakeA_list), grid_dim)
    ax3.imshow(fakeA_grid)
    ax4 = fig.add_subplot(2, 2, 4)
    fakeB_grid = square_grid(np.stack(fakeB_list), grid_dim)
    ax4.imshow(fakeB_grid)
    plt.savefig(os.path.join(model_results_image_dir,'grid.png'))

if __name__ == "__main__":
    save_image_grids(sys.argv[1])