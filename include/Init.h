#pragma once

#include <deque>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "eigen3/Eigen/Dense"
#include "Triangulate.h"

#include <pangolin/display/display.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/image/image_utils.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/viewport.h>
#include <pangolin/utils/range.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void FileRead(std::deque<std::string>& v, std::ifstream &fin);
void MakeTextFile(std::ofstream& fout, const int& imageNum);
void GTPoseRead(std::vector<cv::Vec3f>& v, std::ifstream& fin);



namespace Viewer
{
    class MyVisualize
	{
        private:
            int window_width;
            int window_height;

        public:
            float window_ratio;

            //생성자
            MyVisualize(int width,int height);
            void initialize();
            void active_cam();

            // pts1: GT Pose, pts2: Pose, pts3: 3D Points, pts4: FOV of 3D Points
            void DrawPoint(const std::vector<cv::Vec3d>& tvec, 
                            const std::vector<cv::Vec3f>& gtPose,
                            const std::vector<mvo::Triangulate>& allOfPoints, 
                            const std::vector<cv::Point3f>& fovPoints);
            // circle is before, rectangle is after
            cv::Mat DrawFeatures(cv::Mat& src, std::vector<cv::Point2f>& beforePoints, std::vector<cv::Point2f>& afterPoints);
    };
}//namespace Viewer

namespace mvo
{
    class Initializer
    {
    public:

        // Fix the reference frame
        Initializer(const cv::Mat& ReferenceFrame, float sigma = 1.0, int iterations = 200);

        // Computes in parallel a fundamental matrix and a homography
        // Selects a model and tries to recover the motion and the structure from motion
        bool Initialize(const cv::Mat& CurrentFrame, const std::vector<int> &vMatches12,
                        cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);


        float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &vbMatchesInliers, float sigma);

        float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);

        void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

        int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                        const std::vector<cv::DMatch> &vMatches12, std::vector<bool> &vbInliers,
                        const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

        void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


        // Keypoints from Reference Frame (Frame 1)
        std::vector<cv::KeyPoint> mvKeys1;

        // Keypoints from Current Frame (Frame 2)
        std::vector<cv::KeyPoint> mvKeys2;

        // Current Matches from Reference to Current
        std::vector<cv::DMatch> mvMatches12;
        std::vector<bool> mvbMatched1;

        // Calibration
        cv::Mat mK;

        // Standard Deviation and Variance
        float mSigma, mSigma2;

        // Ransac max iterations
        int mMaxIterations;

        // Ransac sets
        std::vector<std::vector<size_t> > mvSets;
    };
}