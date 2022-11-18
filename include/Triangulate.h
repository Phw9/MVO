#pragma once

#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "Feature.h"

namespace mvo
{
    class Triangulate
    {
    public:
        Triangulate();
        bool CalcWorldPoints(const cv::Mat& pose1,
                            const cv::Mat& pose2,
                            const mvo::Feature& pts1,
                            const mvo::Feature& pts2);
        bool ScalingPoints();
        void MatToPoints3f();
    public:
        cv::Mat mworldMapPoints;
        std::vector<cv::Point3f> mworldMapPointsV;
        std::vector<std::vector<uchar>> mvdesc;
        cv::Vec3d mrvec;
        cv::Vec3d mtvec;
    };
}

bool ManageMapPoints(const std::vector<uchar>& mstatus, std::vector<cv::Point3f>& map);
bool ManageMinusLocal(std::vector<mvo::Feature>& localTrackPoints, const std::vector<int>& id);
bool ManageMinusZ(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id);
void ManageInlier(std::vector<mvo::Feature>& features2d, std::vector<cv::Point3f>& mapPoints3d, const cv::Mat& inlier);
bool ManageMinusZSFM(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id);