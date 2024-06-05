This code utilizes the basic feature classifier algorithm known as Histogram of Oriented Gradient - proposed by N.Dalal and Bill Triggs in 2005. In which they showed that the histogram of oriented gradients
of blocks in the images results in better feature extraction for human detection. 

The basic methodolgy expressed as:
1. compute the gradients in x and y directions.
2. compute the resultant gradient for each pixel in image and their angle  i.e. transform the x and y direction gradients (cartesian plane) to polar plane.
3. divide the image into blocks i.e. (8px by 8px), and for each block the histogram is computed with the bin ranges from 0 - 160 degrees with step size of 20 degrees.
   "" Histogram is the conventional techniques used to find the occurence probability of the particular event in the given range ""
   Thus for 8x8 block the magnitude with angle of each pixel is checked against the angles in the histogram. If the magnitude with the angle resides in the angle of histogram bin then the magnitude is added
   to the bin. If magnitude with angle lie between two bins then the weights of magnitude is calculated which are to be added to the bins.

   for eg:
         i. for the magnitude with angle resides between two bins
             *   angle = 25 degree  resides between 20 and 40 degree bins
                 ------>    for lower bin (20 degree)  ------   ((upper_bin_angle - angle)/bin_size) * magnitude
                 ------>    for upper bin (40 degree)  ------   ((angle - lower_bin_angle)/bin_size) * magnitude
                 For the given condition it can be concluded that the given angle is closer to the lower bin and far apart from the upper bin. Thus the weight of magnitude for lower bin is 0.75 and for upper bin is 0.25, which
                 is also visualized from the expressions that the difference of angles for lower bin is greater than the angle difference of upper bin where the bin_size remains the same i.e. 20 degree.   
4. Then the gradient vectors of each histograms for 8x8 block are ploted on the image, where the vectors shows the strength of gradient and its direction as represented by the length and orientation.
5. The feature descriptors are then evaluated by forming a sliding window over 4 neighboring block - thus forming a cell of size (16x16).
    ""  for image resolution (64x64) --- block size=8
        blocks in row ----> 64/patch_size = 8
        blocks in col ----> 64/patch_size = 8

        16x16 cells
        cells in row  ----> patches in row -1 = 7
        cells in col  ----> patches in col - 1 = 7

        Total cells are ---->  cells in row * cells in col = 7x7 = 49

        Each 16x16 cell contain a vector of HOG having dimension --> 36x1

        Thus total features in the images are ---> Total cells x HoG vector in the cell = 49 x 36 = 1764 features (for the given image)
    ""  
6. the feature descriptors of the images are then used train the model to classify the images based on there local features. For model training i have used the linear support vector machine (SVM)
   The model shows 81% accuracy

   Sample HOG images, where left-sided image show the transformed gray level image, the center image is the contrasted image and the right sided image is the HOG of the contrasted image
   i.
   ![image](https://github.com/RajaAhsan97/Image-classification-using-Histogram-of-Oriented-Gradient-HOG-/assets/155144523/8c24d40c-e361-412f-a3f1-88e13fb91537)

   ii. ![image](https://github.com/RajaAhsan97/Image-classification-using-Histogram-of-Oriented-Gradient-HOG-/assets/155144523/c898203c-fd9a-45e2-853b-48f5efb63ef6)

   iii. ![image](https://github.com/RajaAhsan97/Image-classification-using-Histogram-of-Oriented-Gradient-HOG-/assets/155144523/7951bf4a-e006-4185-90ec-36ba01ee9417)

   iv. ![image](https://github.com/RajaAhsan97/Image-classification-using-Histogram-of-Oriented-Gradient-HOG-/assets/155144523/31def200-ef8a-4783-be64-b0ab871aaec4)



