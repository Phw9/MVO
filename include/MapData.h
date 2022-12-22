#pragma once

#include <algorithm>
#include "ceres/ceres.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "Triangulate.h"
#include "PoseEstimation.h"

#define DISTANCEDESC 20

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
        cv::Mat mdesc;
        std::vector<std::vector<DTYPE>> mvdesc;
        std::vector<int> mindex;
    };

    class Covisibilgraph
    {
    public:
        Covisibilgraph()=default;
        Covisibilgraph(const std::vector<mvo::MapData>& v,
                        const double& focal, const double& ppx, 
                                            const double& ppy);
        ~Covisibilgraph()=default;
        
        template <typename T> void MakeEdgeProj(int gD);
        void MakeEdgeDesc(int gD, mvo::Feature& before, mvo::Triangulate& mapPoints);
        void CullingNode(int gD);

        const std::vector<mvo::MapData>& mglobalMapData;
        // vector < pair <(gD-1)index,(gD)index>
        std::vector<std::vector<std::pair<int, int>>> mgraph;
        const double& mfocal;
        const double& mppx;
        const double& mppy;
    };

    void PushData(std::vector<mvo::MapData>& v, const mvo::MapData& md);
}

