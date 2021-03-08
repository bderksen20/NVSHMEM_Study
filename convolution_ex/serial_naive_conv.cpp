/*  Bill Derksen    3/2021
* 
*   Naive Image Convolution - Sobel Edge Detection
*   - Perform convolution / series of convolutions on Stella image
*   - Goal of this excercise is to better understand convolution process and algorithms
*   - Input image is 512x512 3 channel image with bit depth of 24
*   - OpenCV used for image loading and data structures
* 
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

//std
#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#define image_path "../stella.jpg"

using namespace cv;
int main() {

    // Load image
    int kernel_size = 3;
    Mat image = imread(image_path, IMREAD_COLOR);
    
    if (image.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return -1;
    }

    int im_size = image.size().width;
    int n_pixels = im_size * im_size;
    int im_chan = image.channels();

    // Init kernels
    float sobelx_kernel_data[9] = { -1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1 };

    float sobely_kernel_data[9] = { -1, -2, -1,
                                     0, 0, 0,
                                     1, 2, 1 };

    Mat sobel_x_kernel = Mat(3, 3, CV_32F, sobelx_kernel_data);
    Mat sobel_y_kernel = Mat(3, 3, CV_32F, sobely_kernel_data);

    // Print prelim info about image / matrix
    printf("Image size is %d x %d\n", im_size, im_size);
    printf("Channels: %d\n", im_chan);
    printf("Image continuity: %d\n", image.isContinuous());
    printf("Image matrix type: %s\n", typeToString(image.type()).c_str());
    printf("Image stride: %d\t (n values in each image row, = n channels * image width)\n", (int)image.step);
    printf("\nBeginning convolution process...\n");

    // Pad image...
    copyMakeBorder(image, image, 1, 1, 1, 1, BORDER_CONSTANT, 0);
   
    // Split input layers and convert to 32 bit floats (for applying conv.)
    Mat im_bgr[3];
    split(image, im_bgr);
    im_bgr[0].convertTo(im_bgr[0], CV_32F);
    im_bgr[1].convertTo(im_bgr[1], CV_32F);
    im_bgr[2].convertTo(im_bgr[2], CV_32F);

    // Init output channel vectors
    float* conv_b = new float[n_pixels];
    float* conv_g = new float[n_pixels];
    float* conv_r = new float[n_pixels];

    Mat window;
    int count = 0;

    /* Convolute with Sobel X direction kernel */
    auto start = std::chrono::high_resolution_clock::now();     /* START TIMER */
    for (int col = 0; col < image.cols - 2; col += 1) {
        for (int row = 0; row < image.rows - 2; row += 1) {

            // For each channel.... 
            //   - identify window to apply kernel to
            //   - elementwise mult with kernel and sum
            //   - store result in respective vectors

            window = im_bgr[0](Rect(row, col, kernel_size, kernel_size));   // convolute blue channel window, rect specifes the submat "window"
            conv_b[count] = sum(window.mul(sobel_x_kernel))[0];

            window = im_bgr[1](Rect(row, col, kernel_size, kernel_size));   // convolute green channel window
            conv_g[count] = sum(window.mul(sobel_x_kernel))[0];

            window = im_bgr[2](Rect(row, col, kernel_size, kernel_size));   // convolute red channel window
            conv_r[count] = sum(window.mul(sobel_x_kernel))[0];
            
            // Some debugging prints
            //std::cout << "Computing for row " << row << " col " << col << "\n"
            //std::cout << "Window to convolute: \n" << window << "\n";
            //std::cout << "Kernel: \n" << kernel << "\n";
            //std::cout << "Elementwise mult: \n" << window.mul(kernel) << "\n";
          
            count++;
        }
    }
    
    /* Reshape results and pad: USE IF APPLYING MULTIPLE FILTERS SEQUENTIALLY
    //count = 0;
    //im_bgr[0] = Mat(im_size, im_size, CV_32F, conv_b);
    //im_bgr[1] = Mat(im_size, im_size, CV_32F, conv_g);
    //im_bgr[2] = Mat(im_size, im_size, CV_32F, conv_r);
    //copyMakeBorder(im_bgr[0], im_bgr[0], 1, 1, 1, 1, BORDER_CONSTANT, 0);
    //copyMakeBorder(im_bgr[1], im_bgr[1], 1, 1, 1, 1, BORDER_CONSTANT, 0);
    //copyMakeBorder(im_bgr[2], im_bgr[2], 1, 1, 1, 1, BORDER_CONSTANT, 0);
    */

    /* Convolute with Sobel Y kernel (and sum with previous results) */
    count = 0;
    for (int col = 0; col < image.cols - 2; col += 1) {
        for (int row = 0; row < image.rows - 2; row += 1) {
    
            window = im_bgr[0](Rect(row, col, kernel_size, kernel_size));   // convolute blue channel window
            conv_b[count] += sum(window.mul(sobel_y_kernel))[0];
    
            window = im_bgr[1](Rect(row, col, kernel_size, kernel_size));   // convolute green channel window
            conv_g[count] += sum(window.mul(sobel_y_kernel))[0];
    
            window = im_bgr[2](Rect(row, col, kernel_size, kernel_size));   // convolute red channel window
            conv_r[count] += sum(window.mul(sobel_y_kernel))[0];
    
            count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();   /* STOP TIMER */
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-6;
    
    // Convert b g r outpur vectors to matrix channel form and merge
    Mat conv_image;
    Mat conv_bgr[3];
    Mat(im_size, im_size, CV_32F, conv_b).convertTo(conv_bgr[0], CV_8UC1);
    Mat(im_size, im_size, CV_32F, conv_g).convertTo(conv_bgr[1], CV_8UC1);
    Mat(im_size, im_size, CV_32F, conv_r).convertTo(conv_bgr[2], CV_8UC1);

    // Merge channel layers: use array and size or vector
    merge(conv_bgr, 3, conv_image);

    std::cout << "Convolution completed successfully with execution time of: " << std::setprecision(9) << time_taken << " ms\n";

    // Display and write image
    imshow("Window", conv_image);
    int k = waitKey(0); // Wait for a keystroke in the window

    //imwrite("../output_images/conv_out.jpg", conv_image);

    delete[] conv_b;
    delete[] conv_g;
    delete[] conv_r;

    return 0;
}

