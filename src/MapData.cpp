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
    if(mpoint2D.size() != ft.mfeatures.size()) return false;
    return true;
}

bool mvo::MapData::Get3DPoints(const mvo::Triangulate& tr)
{
    mpoint3D.clear();
    mvdesc.clear();
    mvdesc = tr.mvdesc;
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