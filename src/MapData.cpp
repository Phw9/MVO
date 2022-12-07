#include "MapData.h"

mvo::MapData::MapData()
{
    mpoint2D.clear();
    mpoint3D.clear();
    mvdesc.clear();
    mglobalrvec.zeros();
    mglobaltvec.zeros();
}

bool mvo::MapData::GetSFMPose(const mvo::StrctureFromMotion& sfm)
{
    mglobalRTMat = sfm.mCombineRt.clone();  mglobalRMat = sfm.mRotation.clone();
    mglobalrvec = sfm.mrvec;        mglobaltvec = sfm.mtvec;
    return true;
}

bool mvo::MapData::GetPnPPose(const mvo::PoseEstimation& pe)
{
    mglobalRTMat = pe.mCombineRt.clone();  mglobalRMat = pe.mRotation.clone();
    mglobalrvec = pe.mrvec;  mglobaltvec = pe.mtvec;  mglobalTranslation = pe.mtranslation;
    // std::cout << "mglobalRTMat: " << mglobalRTMat <<std::endl;
    // std::cout << "pe.mCombineRT: " << pe.mCombineRt <<std::endl;
    return true;
}

bool mvo::MapData::Get2DPoints(const mvo::Feature& ft)
{
    mpoint2D.clear();
    mpoint2D = ft.mfeatures;
    mdesc = ft.mdesc;
    if(mpoint2D.size() != ft.mfeatures.size()) return false;
    return true;
}

bool mvo::MapData::Get3DPoints(const mvo::Triangulate& tr)
{
    mpoint3D.clear();
    mpoint3D = tr.mworldMapPointsV;
    if(mpoint3D.size() != tr.mworldMapPointsV.size()) return false;
    // std::cout << "Get3DPoints: " << mpoint3D.size() << ", " << tr.mworldMapPointsV.size() << std::endl;
    // std::cout << "Get3Dmvdesc: " << mvdesc.size() << ", " << tr.mvdesc.size() << std::endl;

    return true;
}

void mvo::PushData(std::vector<mvo::MapData>& v, const mvo::MapData& md)
{
    mvo::MapData temp;

    temp.mglobalRMat = md.mglobalRMat.clone();
    temp.mglobalRTMat = md.mglobalRTMat.clone();
    temp.mglobalrvec = md.mglobalrvec;
    temp.mglobaltvec = md.mglobaltvec;
    temp.mpoint2D = md.mpoint2D;
    temp.mpoint3D = md.mpoint3D;
    temp.mvdesc = md.mvdesc;
    v.push_back(temp);
}

mvo::Covisibilgraph::Covisibilgraph(const std::vector<mvo::MapData>& v,
                                 const double& focal, const double& ppx, 
                                                      const double& ppy)
                                : mglobalMapData(v), mfocal(focal), mppx(ppx), mppy(ppy){};

template <typename T> void mvo::Covisibilgraph::MakeEdgeProj(int gD)
{
    const T theta = sqrt(mglobalMapData.at(gD).mglobalrvec[0] * mglobalMapData.at(gD).mglobalrvec[0] +
                         mglobalMapData.at(gD).mglobalrvec[1] * mglobalMapData.at(gD).mglobalrvec[1] +
                         mglobalMapData.at(gD).mglobalrvec[2] * mglobalMapData.at(gD).mglobalrvec[2]);

    const T tvec_eig_0 = mglobalMapData.at(gD).mglobaltvec[0];
    const T tvec_eig_1 = mglobalMapData.at(gD).mglobaltvec[1];
    const T tvec_eig_2 = mglobalMapData.at(gD).mglobaltvec[2];

    const T w1 = mglobalMapData.at(gD).mglobalrvec[0] / theta;
    const T w2 = mglobalMapData.at(gD).mglobalrvec[1] / theta;
    const T w3 = mglobalMapData.at(gD).mglobalrvec[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    Eigen::Matrix<T, 3, 4> worldToCam;
    worldToCam << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 3, 1> pixel3d;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << mfocal, 0, mppx,
        0, mfocal, mppy,
        0, 0, 1;
    
    pixel3d = Kd.cast<T>() * worldToCam * mglobalMapData.at(gD).mpoint3D.at(gD);
}

void mvo::Covisibilgraph::MakeEdgeDesc(int gD, mvo::Feature& before, mvo::Triangulate& mapPoints)
{
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matches.reserve(2000);
    std::vector<std::pair<int,int>> temp;
    temp.reserve(2000);
    std::pair<int,int> idxMatch;

    /*
    1. Matching 2D Points between before(query) and gD(train) 
    2. Correspond 3dPoints between (before==mapPoints) and (gD-1)
    3. Correcting index (before) in (gD-1)
    */

    matcher.match(before.mdesc, mglobalMapData.at(gD).mdesc, matches);

    int max=0, min=100, sum=0;
    int maxq=0; int maxt=0;
    int inlier=0; int inlier2=0;
    int N=matches.size();

    for(int i=0; i<N; i++)
    {
        if(max < matches.at(i).distance) max = matches.at(i).distance;
        if(maxq < matches.at(i).queryIdx) maxq = matches.at(i).queryIdx;
        if(maxt < matches.at(i).trainIdx) maxt = matches.at(i).trainIdx;
        if(min > matches.at(i).distance) min = matches.at(i).distance;
        sum += matches.at(i).distance;

        if(matches.at(i).distance<1)
        {
            inlier++;
            idxMatch.first=matches.at(i).queryIdx; idxMatch.second=matches.at(i).trainIdx;
            temp.emplace_back(std::move(idxMatch));
        }
    }

    // if mapPoints.at(temp.first) == (gD-1)3dpoints
    // temp.first = (gD-1) 3dPoints idx
    for(int i=0; i<temp.size(); i++)
    {
        for(int j=0; j<mglobalMapData.at(gD-1).mpoint3D.size(); j++)
        {
            if(mapPoints.mworldMapPointsV.at(temp.at(i).first).x == mglobalMapData.at(gD-1).mpoint3D.at(j).x
            && mapPoints.mworldMapPointsV.at(temp.at(i).first).y == mglobalMapData.at(gD-1).mpoint3D.at(j).y
            && mapPoints.mworldMapPointsV.at(temp.at(i).first).z == mglobalMapData.at(gD-1).mpoint3D.at(j).z)
            {
                inlier2++;
                temp.at(i).first = j;
                break;
            }
        }
    }

    mgraph.emplace_back(std::move(temp));
    std::cout << "mgraph size: " << mgraph.at(gD-1).size() << std::endl;;
    std::cout << "Query: " << maxq << ", Train: " << maxt << " Avg Distance: " << (float)sum/N << std::endl;
    std::cout << "inlier: " << inlier << ", inlier2: " << inlier2 <<std::endl;
}