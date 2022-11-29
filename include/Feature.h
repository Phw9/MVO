#pragma once

#include <vector>
#include "eigen3/Eigen/Dense"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/video.hpp"
#include "opencv2/core.hpp"


#define DTYPE uchar
#define ERRORSIZE 25
#define MAXCORNER 1000
#define QUALITYLEVEL 0.001
#define MINDISTANCE 8


namespace mvo
{
    class Feature
    {
    public:
        Feature();
        ~Feature()=default;
        
        bool CornerFAST(const cv::Mat& src);
        bool GoodFeaturesToTrack(const cv::Mat& src);
        bool OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next);
        std::vector<uchar> mstatus;
        std::vector<float> merr;
        std::vector<cv::Point2f> mfeatures;
        std::vector<std::vector<DTYPE>> mvdesc;
    };
}//namespace mvo

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before);
bool KeyPointToVec(const std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& features2d);
bool VecToKeyPoint(const std::vector<cv::Point2f>& features2d, std::vector<cv::KeyPoint>& kp);
bool MatToVec(const cv::Mat m, std::vector<std::vector<DTYPE>>& v);