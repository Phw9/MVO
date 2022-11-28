#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "opencv2/core.hpp"
#include "ceres/ceres.h"
#include "ceres/loss_function.h"
#include "Triangulate.h"
#include "MapData.h"

// static float camXf = 6.018873000000e+02; 
// static float camYf = 1.831104000000e+02; 
// static float focalLengthf = 7.070912000000e+02; 
static double camX = 6.018873000000e+02;
static double camY = 1.831104000000e+02;
static double focald = 7.070912000000e+02;

namespace mvo
{
    struct SnavelyReprojectionError
    {
        SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d worldHomoGen4d, 
                                 double focal, double ppx, double ppy);
        ~SnavelyReprojectionError() = default;

        template <typename T> bool operator()(const T *const rvec_eig,
                                              const T *const tvec_eig,
                                              T *residuals) const;

        double observed_x;
        double observed_y;
        const Eigen::Vector4d worldHomoGen4d;
        double focal;
        double ppx;
        double ppy;
    };

    class BundleAdjustment
    {
    public:
    BundleAdjustment();
    BundleAdjustment(mvo::Feature& ft, mvo::PoseEstimation& rt,
                     mvo::Triangulate& tri);
    BundleAdjustment(mvo::MapData& data);
    ~BundleAdjustment() = default;

    bool MotionOnlyBA();
    bool LocalBA();
    bool FullBA();

    mvo::Feature ft;
    mvo::PoseEstimation rt;
    mvo::Triangulate tri;
    mvo::MapData data;
    };

    bool MotionBA();
}