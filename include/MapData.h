#pragma once

#include "Triangulate.h"
#include "PoseEstimation.h"



namespace mvo
{
    class MapData
    {
    public:
        MapData();
        ~MapData()=default;
        
        bool GetSFMPose(const mvo::StrctureFromMotion& sfm);
        bool GetPnPPose(const mvo::PoseEstimation& pe);
        bool Get2DPoints(const mvo::Feature& feature);
        bool Get3DPoints(const mvo::Triangulate& tr);

        std::vector<cv::Point2f> mpoint2D;
        std::vector<cv::Point3f> mpoint3D;
        cv::Mat mglobalRTMat;
        cv::Mat mglobalRMat;
        cv::Vec3d mglobalrvec;
        cv::Vec3d mglobaltvec;
        cv::Vec3d mglobalTranslation;
        std::vector<std::vector<DTYPE>> mvdesc;
    };

    void PushData(std::vector<mvo::MapData>& v, const mvo::MapData& md);
}