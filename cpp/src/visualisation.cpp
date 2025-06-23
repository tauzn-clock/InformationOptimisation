#ifndef VISUALISATION_H
#define VISUALISATION_H

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat visualisation(cv::Mat mask, int mask_max) {

    std::vector<cv::Vec3b> colors(mask_max);
    colors[0] = cv::Vec3b(0, 0, 0); // background
    for (int i = 1; i <mask_max; i++)
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);

    cv::Mat output(mask.size(), CV_8UC3);
    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            int label = (int)mask.at<unsigned int>(r, c);
            output.at<cv::Vec3b>(r, c) = colors[label];
        }
    }
    return output;
}
#endif