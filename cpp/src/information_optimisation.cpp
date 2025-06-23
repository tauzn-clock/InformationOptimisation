#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <random>
#include "information_optimisation.h"

using namespace std;

array<float,4> get_plane(array<float,3> a, array<float,3> b, array<float,3> c) {
    array<float,3> v1, v2;

    v1[0] = b[0] - a[0];
    v1[1] = b[1] - a[1];
    v1[2] = b[2] - a[2];

    v2[0] = c[0] - a[0];
    v2[1] = c[1] - a[1];
    v2[2] = c[2] - a[2];

    array<float,4> plane;

    plane[0] = v1[1] * v2[2] - v1[2] * v2[1];
    plane[1] = v1[2] * v2[0] - v1[0] * v2[2];
    plane[2] = v1[0] * v2[1] - v1[1] * v2[0];

    float norm = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]) + 1e-10;
    plane[0] /= norm;
    plane[1] /= norm;
    plane[2] /= norm;

    plane[3] = -plane[0] * a[0] - plane[1] * a[1] - plane[2] * a[2];

    if (plane[2] < 0) {
        plane[0] = -plane[0];
        plane[1] = -plane[1];
        plane[2] = -plane[2];
        plane[3] = -plane[3];
    }

    return plane;
}

int information_optimisation(cv::Mat depth, YAML::Node config, int max_plane, vector<int>& input_mask) {

    float fx = config["camera_params"]["fx"].as<float>();
    float fy = config["camera_params"]["fy"].as<float>();
    float cx = config["camera_params"]["cx"].as<float>();
    float cy = config["camera_params"]["cy"].as<float>();
    float scale = config["scale"].as<float>();

    float R = config["R"].as<float>();
    float eps = config["eps"].as<float>();
    float STATES = log(R/eps);

    float conf = config["conf"].as<float>();
    float inlier_th = config["inlier_th"].as<float>();

    float ITERATION = log(1 - conf) / log(1 - pow(inlier_th, 3));
    cout<<ITERATION<<endl;

    int H = depth.rows;
    int W = depth.cols;

    // Assert size of mask is H*W
    if (input_mask.size() != (long unsigned int) H*W) {
        cerr << "Mask size does not match depth size" << endl;
        return {};
    }

    vector<float> sigma(H*W);
    vector<float> log_sigma(H*W);

    vector<array<float,3> > points(H*W);
    vector<array<float,3> > direction_vector(H*W);
    vector<int> mask(H*W, 0);
    int total_points = 0;

    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            points[i*W + j][0] = (j - cx) * depth.at<unsigned short>(i, j) / fx / scale;
            points[i*W + j][1] = (i - cy) * depth.at<unsigned short>(i, j) / fy / scale;
            points[i*W + j][2] = depth.at<unsigned short>(i, j) / scale;

            float norm = sqrt(points[i*W + j][0] * points[i*W + j][0] + points[i*W + j][1] * points[i*W + j][1] + points[i*W + j][2] * points[i*W + j][2]) + 1e-10;
            direction_vector[i*W + j][0] = points[i*W + j][0] / norm;
            direction_vector[i*W + j][1] = points[i*W + j][1] / norm;
            direction_vector[i*W + j][2] = points[i*W + j][2] / norm;

            //Sigma function
            sigma[i*W + j] = 0.01 * depth.at<unsigned short>(i, j) / scale;
            log_sigma[i*W + j] = log(sigma[i*W + j]) - log(eps) + 0.5 * log(2 * 3.14159) - STATES;

            if (depth.at<unsigned short>(i, j) == 0 || input_mask[i*W + j]==0) {
                mask[i*W + j] = -1;
            }
            else{
                total_points++;
            }
        }
    }

    vector<float> information(max_plane, 0);
    vector<array<float,4> > plane(max_plane);
    vector<int> available_points(H*W, 0);
    int available_points_cnt = 0;

    information[0] = total_points * STATES;

    array<float,4> trial_plane;
    vector<float> trial_error(H*W, 0);
    float trial_total_error = 0;

    vector<float> best_error(H*W, 0);
    float best_total_error;

    for (int plane_cnt = 1; plane_cnt < max_plane; plane_cnt++) {
        auto start = chrono::high_resolution_clock::now();

        available_points_cnt = 0;
        for (int i = 0; i < H*W; i++) {
            if (mask[i] == 0) {
                available_points[available_points_cnt] = i;
                available_points_cnt++;
            }
        }
        available_points.resize(available_points_cnt);

        if (available_points_cnt <=3){
            for (int i=plane_cnt; i<max_plane; i++){
                information[i] = information[i-1];
            }
            break;
        }
        
        information[plane_cnt] = information[plane_cnt-1] + total_points * log((float)(plane_cnt+1)/plane_cnt) + 3 * STATES;

        best_total_error = 0;

        std::random_device rd;  // Obtain a seed from the hardware
        std::mt19937 gen(rd()); // Initialize a Mersenne Twister engine
        std::uniform_int_distribution<> distrib(1, available_points_cnt); // Range from 1 to 100

        for (int trial=0; trial<ITERATION; trial++){

            trial_total_error = 0;

            int index_a = distrib(gen);
            int index_b = distrib(gen);
            int index_c = distrib(gen);

            if (index_a == index_b || index_a == index_c || index_b == index_c) {
                continue;
            }

            trial_plane = get_plane(points[available_points[index_a]], points[available_points[index_b]], points[available_points[index_c]]);

            for(const int& j : available_points) {
                trial_error[j] = - trial_plane[3] / (trial_plane[0] * direction_vector[j][0] + trial_plane[1] * direction_vector[j][1] + trial_plane[2] * direction_vector[j][2] + 1e-10)*direction_vector[j][2];
                trial_error[j] = (trial_error[j] - points[j][2]) / sigma[j];
                trial_error[j] = pow(trial_error[j], 2)/2 + log_sigma[j];

                if (trial_error[j] < 0) {
                    trial_total_error += trial_error[j];
                }
            }

            if (trial_total_error < best_total_error) {
                swap(best_error, trial_error);
                best_total_error = trial_total_error;
                plane[plane_cnt] = trial_plane;
            }
        }

        information[plane_cnt] += best_total_error;
        for(const int& j : available_points) {
            if (best_error[j] < 0) {
                mask[j] = plane_cnt;
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout<<duration.count()<<endl;
    }

    int min_information = 0;
    for (int i = 0; i < max_plane; i++) {
        if (information[i] < information[min_information]) {
            min_information = i;
        }
    }

    cout<<"Min Info: "<<min_information<<" "<<information[min_information]<<endl;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            input_mask[i*W+j] = mask[i*W + j];
        }
    }

    return min_information;
}