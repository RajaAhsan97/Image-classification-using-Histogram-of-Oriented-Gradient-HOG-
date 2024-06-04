"""
    COMMENT:
    need further improvement for adjustment of image sharpening to optimum level,
    which is then used by Sobel kernel compute the gradient mangnitude and angle.

    see it tommorow
"""

"""
    Description of HOG feature vectors
    
    1. scaled image (4X)
    For this i have to compute offsets in the array to create a cell og four 8x8 of image (i.e. four 9x1 HOGs [36]) and compute L2 normalization
    in my case there will by 31 cells horizontally and 31 cells vertically, and each cell will have 36 features. This 31x31x36=34596 features.

    2. original resolution of image (64x64) --- patch size=8
        patches in row ----> 64/patch_size = 8
        patches in col ----> 64/patch_size = 8

        16x16 cells
        cells in row  ----> patches in row -1 = 7
        cells in col  ----> patches in col - 1 = 7

        Total cells are ---->  cells in row * cells in col = 7x7 = 49

        Each 16x16 cell contain a vector of HOG having dimension --> 36x1

        Thus total features in the images are ---> Total cells x HoG vector in the cell = 49 x 36 = 1764 features
"""



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
"""
    Get image from directory outside the current working directory
"""

#base_wd = os.path.split(os.getcwd())[0]

##data_dir = os.path.join(base_wd, 'archive')
##
### get image from directory path
##data = os.listdir(data_dir)[0]
##img_path = os.path.join(data_dir, data)


"""
    STEP 2: Compute gradient magnitude and direction
"""

def ComputeGradient(image):
    # compute x and y gradient magnitude
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy= cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    # cartesian to polar plane conversion
    return cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
"""
"""

"""
    STEP 3: Divide image into 8x8 patches
"""

def ComputeHistogram(mag, ang, patch_size, img_resolution):
    patch_pixels_count = patch_size**2
    min_angle, max_angle, bin_size, angle_threshold =  0, 160, 20, 180

    # divide image into 8x8 patches
    x, y = np.arange(0, img_resolution[1]+1, patch_size), np.arange(0, img_resolution[0]+1, patch_size)

    # histogram bins 0-160 degree with step size of 20 degree
    histogram_bins = list(np.arange(min_angle, max_angle+1, bin_size))
    patch_bins = []
    
    for idx_y in range(len(y)-1):
        for idx_x in range(len(x)-1):        
            magnitude_patch = mag[y[idx_y]:y[idx_y+1], x[idx_x]:x[idx_x+1]]
            angle_patch = ang[y[idx_y]:y[idx_y+1], x[idx_x]:x[idx_x+1]]
            
            mag_reshape = magnitude_patch.reshape(1, patch_pixels_count)
            ang_reshape = angle_patch.reshape(1,patch_pixels_count)

            patch_bins_aggr = list(np.zeros(len(histogram_bins)))

            # iterate over pixels in the patch
            """
                0 - 160 degree bins with step size of 20 degree
            """
        
            for iterate in range(np.shape(mag_reshape)[1]):
                mag_angle_px = ang_reshape[0][iterate]
                mag_px = mag_reshape[0][iterate]
                if mag_angle_px in histogram_bins:
                    bin_index = histogram_bins.index(mag_angle_px)
                    patch_bins_aggr[bin_index] += mag_px
                else:
                    if mag_angle_px == angle_threshold:
                        patch_bins_aggr[0] += mag_px
                    elif mag_angle_px > histogram_bins[-1] and mag_angle_px < angle_threshold:
                        patch_bin_val_lower, patch_bin_val_upper = ((180 - mag_angle_px)/bin_size) * mag_px, ((mag_angle_px - histogram_bins[-1])/bin_size) * mag_px
                        patch_bins_aggr[-1] += patch_bin_val_lower
                        patch_bins_aggr[0] += patch_bin_val_upper
                    else:
                        for indx in range(len(histogram_bins)-1):
                            if mag_angle_px > histogram_bins[indx] and mag_angle_px < histogram_bins[indx+1]:
                                patch_bin_val_lower, patch_bin_val_upper = ((histogram_bins[indx+1] - mag_angle_px)/bin_size) * mag_px, ((mag_angle_px - histogram_bins[indx])/bin_size) * mag_px 
                                patch_bins_aggr[indx] += patch_bin_val_lower
                                patch_bins_aggr[indx+1] += patch_bin_val_upper
                                break
            patch_bins.append(patch_bins_aggr)
    return patch_bins, histogram_bins
        
"""
"""

"""
    to plot gradient vectors on image
"""
def ComputeGradientVectors(image_nm, gray_img, sharpen_img, patch_hist_bins, patch_size, img_resolution, histogram):
    global images
    centroid_y, centroid_x = int(patch_size/2), int(patch_size/2)
    hist_img = np.zeros((img_resolution[0], img_resolution[1]), dtype=int)

    offset_x, offset_y = 0,0
    r1, r2 = 0, 8
    c1, c2 = 0, 8

    for patch in patch_hist_bins:
        norm_patch = []
        norm_k = np.sqrt(np.sum(patch))
        if norm_k == 0:
            norm_patch = patch
        else:
            norm_patch = np.divide(np.divide(patch, norm_k), 5)

        cell_matrix = np.zeros((patch_size, patch_size), dtype=int)
        for hist_bin in range(np.shape(patch)[0]):
            angle = histogram[hist_bin]
            cosine_comp = math.cos(math.radians(angle))
            sine_comp = math.sin(math.radians(angle))
            
            xx = norm_patch[hist_bin]*cosine_comp
            yy = norm_patch[hist_bin]*sine_comp
            
            x2 = round(centroid_x + xx)
            y2 = round(centroid_y - yy)
            
            cv2.line(cell_matrix, (centroid_x, centroid_y), (x2,y2), norm_patch[hist_bin], 1)
            hist_img[r1+offset_y:r2+offset_y, c1+offset_x:c2+offset_x] = cell_matrix

        offset_x += 8
        if offset_x == img_resolution[1]:
            offset_y += 8
            offset_x = 0
            if offset_y == img_resolution[0]:
                break
    Plot(1,3,image_nm,Gray_IMG = gray_img, Sharp_IMG = sharpen_img, Hist_IMG = hist_img)

"""

"""
def IndexMatrix(patch_bins, img_res, patch_size):
    # forming index matrix to store HOG computed for each 8x8 patch
    hist_row, hist_col = int(img_res[1]/patch_size), int(img_res[0]/patch_size)
    hist_idx_mat = np.zeros((hist_row, hist_col), dtype=int)

    row_idx , col_idx = 0, 0
    for index in range(len(patch_bins)):
        hist_idx_mat[row_idx, col_idx] = index
        col_idx += 1
        if col_idx == hist_col:
            row_idx += 1
            col_idx = 0
    return hist_idx_mat, hist_row, hist_col

def ComputeFeatureVectors(patch_hist_bins, patch_size, img_resolution):
    index_matrix, mat_row, mat_col = IndexMatrix(patch_hist_bins, img_resolution, patch_size)
    # forming 16x16 cells of HOG and apply L2 normalization

    # creating sliding window of 2x2 over the 8x8 patches i.e forming a 16x16 cell
    cells = []
    normalize_const = []
    for slide_row_idx in range(mat_row-1):
        for slide_col_idx in range(mat_col-1):
            # get HoG of 4 neighboring 8x8 patches in the 16x16 cell
            patch1, patch2, patch3, patch4 = index_matrix[slide_row_idx, slide_col_idx], index_matrix[slide_row_idx, slide_col_idx+1], index_matrix[slide_row_idx+1, slide_col_idx], index_matrix[slide_row_idx+1, slide_col_idx+1]
            # append patches (1x9) to form (1x36) 
            cell = patch_hist_bins[patch1] + patch_hist_bins[patch2] + patch_hist_bins[patch3] + patch_hist_bins[patch4]
            # compute normalization contant
            k = np.sqrt(np.sum(np.power(cell, 2)))
            if k == 0.0:
                cells.extend(list(cell))
            else:
                # normalize (1x36) HoG values
                normalize_cell = np.divide(cell, k)
                cells.extend(list(normalize_cell))
                
            normalize_const.append(k)
            cell = []

    #c = np.array(cells)
    
    return cells


def Plot(row, col, img_name, **kwargs):
    fig = plt.figure(figsize=(20,20))
    keys = list(kwargs.keys())
    
    for k in range(len(keys)):
        ax = fig.add_subplot(row,col,k+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(keys[k])
        ax.imshow(kwargs[keys[k]])

    plt.savefig(img_name)
    plt.close()


"""
    store the computed feature descriptors of all image, which are used for further processing 
"""

"""
    STEP 1: Read image and convert it to gray scale
"""


def HOG(data, data_rslt_pth):
    print("Computing Feature descriptors for Dataset: ", data[0].rsplit('\\')[-2])
    # define patch size for HOG (8x8) for which oriented vectors are to be computed
    patch_size = 8
    images = len(data)

    fd_dict = {}
    image_count = 0

    function_comp_time = {}
    while image_count < images:
        img_path = data[image_count]
        # read image
        rgb_img = plt.imread(img_path)
        rgb_shape = np.shape(rgb_img)

        height, width = int(rgb_shape[0]/patch_size), int(rgb_shape[1]/patch_size)

        RGB_IMAGE = rgb_img[0:height*patch_size, 0:width*patch_size]

        # map rgb image to Gray levels
        gray_frame = cv2.cvtColor(RGB_IMAGE, cv2.COLOR_BGR2GRAY)
        frame_resol = np.shape(gray_frame)

        # image Sharpening kernel
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        contrast_img = cv2.filter2D(gray_frame, -1, kernel)

        magnitude, angle = ComputeGradient(gray_frame)
        #print("MD", magnitude)
        
        start_time_func1 = time.time()
        patch_bins, histogram = ComputeHistogram(magnitude, angle, patch_size, frame_resol)
        end_time_func1 = time.time()

        img_nm = img_path.rsplit("\\")[-1]
        # if image is in .gif format then change extension to .png to save the HOG image
        ext = img_nm.rsplit(".")
        if ext[-1] == "gif":
            ext[-1] = ".png"
            img_nm = "".join(ext)
        image_path = os.path.join(data_rslt_pth, img_nm)
        start_time_func2 = time.time()
        ComputeGradientVectors(image_path, gray_frame, contrast_img, patch_bins, patch_size, frame_resol, histogram)
        end_time_func2 = time.time()

        # Compute feature vectors
        start_time_func3 = time.time()
        feature_descrip = ComputeFeatureVectors(patch_bins, patch_size, frame_resol)
        end_time_func3 = time.time()

        fd_dict[img_nm] = feature_descrip

        Comp_hist_time = end_time_func1 - start_time_func1
        Comp_Grad_vec_time = end_time_func2 - start_time_func2
        Comp_Feat_vec_time = end_time_func3 - start_time_func3

        function_comp_time[img_nm] = {'histogram Func': Comp_hist_time, 'Gradient Vectors Func': Comp_Grad_vec_time, 'Feature Vector Func': Comp_Feat_vec_time}        
            
##        print("--------------------------------------------------------------------------------------")
##        print("Image ", image_count+1, ": ", data[image_count], " ------ Resolution: ", frame_resol)
##        print("Functions Execution Time")
##        print("Function -- ComputeHistogram: ", end_time_func1 - start_time_func1)
##        print("Function -- ComputeGradientVectors: ", end_time_func2 - start_time_func2)
##        print("Function -- ComputeFeatureVectors: ", end_time_func3 - start_time_func3)

        image_count += 1

    HOG_fds = pd.DataFrame.from_dict(fd_dict, orient='index').transpose()
    #HOG_fds.to_csv(data_path)

    return HOG_fds, function_comp_time
"""
    To make a function to plot image or surface plots of a given image frame when called from anywhere in the code
"""
##X, Y = np.meshgrid(np.arange(0, frame_resol[1], 1), np.arange(0, frame_resol[0], 1))
##
##fig = plt.figure()
##
##ax = fig.add_subplot(2,2,1)
##ax.set_title("Test Image (Gray Levels)")
##ax.set_xticks([])
##ax.set_yticks([])
##ax.imshow(gray_frame)
##
##ax = fig.add_subplot(2,2,2, projection="3d")
##ax.set_title("Test Image (Surface plot)")
##ax.set_xticks([])
##ax.set_yticks([])
##ax.view_init(elev=45, azim=45, roll=0)
##ax.plot_surface(X,Y,gray_frame)
##
##ax = fig.add_subplot(2,2,3)
##ax.set_title("Sharpen Test Image")
##ax.set_xticks([])
##ax.set_yticks([])
##ax.imshow(sharp_img_k1)
##
##ax = fig.add_subplot(2,2,4, projection="3d")
##ax.set_title("Sharpen Test Image (Surface plot)")
##ax.set_xticks([])
##ax.set_yticks([])
##ax.view_init(elev=45, azim=45, roll=0)
##ax.plot_surface(X,Y, sharp_img_k1)
##
##plt.show()
