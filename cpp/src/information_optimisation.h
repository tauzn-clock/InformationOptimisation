#ifndef INFORMATION_OPTIMISATION_H
#define INFORMATION_OPTIMISATION_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>

int information_optimisation(cv::Mat, YAML::Node, int, std::vector<int>&);

#endif