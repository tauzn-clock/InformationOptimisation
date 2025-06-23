#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

int get_rgb_regions(cv::Mat img, YAML::Node config, cv::Mat& seg_mask) {
    //Grayscale, Blur , Canny
    cv::Mat edge;
    cv::cvtColor(img, edge, cv::COLOR_BGR2GRAY);
    int kernel_size = config["edge_detection_params"]["guassian_blur"]["kernel_size"].as<int>();
    float sigma = config["edge_detection_params"]["guassian_blur"]["sigma"].as<float>();
    cv::GaussianBlur(edge, edge, cv::Size(kernel_size, kernel_size), sigma);
    float low_threshold = config["edge_detection_params"]["canny"]["low_threshold"].as<float>();
    float high_threshold = config["edge_detection_params"]["canny"]["high_threshold"].as<float>();
    cv::Canny(edge, edge, low_threshold, high_threshold);

    //Get mask
    cv::bitwise_not(edge, seg_mask);
    int n_labels = cv::connectedComponents(seg_mask, seg_mask, 4);

    std::vector<std::array<int,2> > edge_index;
    for(int i=0; i<edge.rows; i++){
        for(int j=0; j<edge.cols; j++){
            if(edge.at<unsigned char>(i,j) == 255){
                std::array<int,2> index = {i,j};
                edge_index.push_back(index);
            }
        }
    }

    cv::Mat new_seg_mask = seg_mask.clone();
    cv::Mat new_edge = edge.clone();
    int edge_size = edge_index.size();
    int new_edge_size;

    std::array< std::array<int,2>, 8> index_array = {
        std::array<int,2>{-1,0},
        std::array<int,2>{1,0},
        std::array<int,2>{0,-1},
        std::array<int,2>{0,1},
        std::array<int,2>{-1,-1},
        std::array<int,2>{-1,1},
        std::array<int,2>{1,-1},
        std::array<int,2>{1,1}
    };

    while (edge_size > 0){
        std::cout << "edge_size: " << edge_size << std::endl;
        new_edge_size = 0;
        for (int i=0; i<edge_size; i++){
            std::array<int,2> index = edge_index[i];
            int x = index[0];
            int y = index[1];

            bool filled = false;
            for (int j=0; j<8; j++){
                int x_ = x + index_array[j][0];
                int y_ = y + index_array[j][1];

                if (x_ >= 0 && x_ < seg_mask.rows && y_ >= 0 && y_ < seg_mask.cols){
                    if (edge.at<unsigned char>(x_,y_) != 255){
                        filled = true;
                        new_seg_mask.at<unsigned int>(x,y) = seg_mask.at<unsigned int>(x_,y_);
                        new_edge.at<unsigned char>(x,y) = 0;
                        break;
                    }
                }
            }

            if (not filled){
                edge_index[new_edge_size] = index;
                new_edge_size++;
            }
        }
        edge_size = new_edge_size;
        seg_mask = new_seg_mask.clone();
        edge = new_edge.clone();
    }

    return n_labels;
}