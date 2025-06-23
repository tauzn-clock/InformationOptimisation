#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include "visualisation.cpp"
#include "get_rgb_regions.cpp"
#include "information_optimisation.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    std::cout << argv[1] << std::endl;
    
    YAML::Node config = YAML::LoadFile(argv[1]);
    
    for(int i=0; i<1449; i++){
        std::string rgb_path = config["file_path"].as<std::string>() + "/rgb/" + std::to_string(i) + ".png";
        std::string depth_path = config["file_path"].as<std::string>() + "/depth/" + std::to_string(i) + ".png";

        cv::Mat img = cv::imread(rgb_path, cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

        int H = depth.rows;
        int W = depth.cols;
        std::vector<int> plane_mask(H*W, 1);

        int plane = information_optimisation(depth, config, 10, plane_mask);

        cv::imwrite(config["file_path"].as<std::string>() + "/our/" +std::to_string(i)+".png", plane_mask);
    }

    //cv::Mat labelImg = visualisation(seg_mask, n_labels);
    //plane_mask.convertTo(plane_mask, CV_32SC1);
    //cv::Mat Img = visualisation(plane_mask, plane_cnt+1);

    //cv::imshow("labelImg", labelImg);
    //cv::imshow("plane_mask", Img);
    //cv::waitKey(0);

    return 0;
}
