#include <iostream>
#include "Feature.h"


mvo::Feature::Feature()
{
    mfeatures.clear();
    mstatus.clear();
    merr.clear();
    mvdesc.clear();
}



// bool mvo::Feature::CornerFAST(const cv::Mat& src)
// {
//     // threshold는 120이고 비최대 억제를 수행한다
//     cv::FAST(src, mfastKeyPoints, 120, true);
//     if (mfastKeyPoints.empty())
//     {
//         std::cerr << "Failed FAST detection" << std::endl;
//         return false;
//     }
//     return true;
// }

bool mvo::Feature::GoodFeaturesToTrack(const cv::Mat& src)
{
    cv::goodFeaturesToTrack(src, mfeatures, MAXCORNER, QUALITYLEVEL, MINDISTANCE);
    if(mfeatures.empty())
    {
        std::cerr << "failed to goodFeaturesToTrack" << std::endl;
        return false;
    }
    else if(mfeatures.size() < 100)
    {
        std::cerr << "tracker size small than 100" << std::endl;
        return false;
    }
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();

    cv::Mat desc1;
    std::vector<cv::KeyPoint> kp;
    VecToKeyPoint(mfeatures, kp);
    std::cout << "before mfeatures: " << mfeatures.size() << std::endl;
    std::cout << "before kp size: " << kp.size() << std::endl;
    brief->compute(src, kp, desc1);
    mfeatures.clear();
    KeyPointToVec(kp,mfeatures);
    mvdesc.clear();
    MatToVec(desc1, mvdesc);
    std::cout << "desc mat size: " << desc1.size() << std::endl;
    std::cout << "after mfeatures: " << mfeatures.size() << std::endl;
    std::cout << "after kp size: " << kp.size() << std::endl;
    std::cout << "mvdesc: " << mvdesc.size() << std::endl;
    return true;
}

bool mvo::Feature::OpticalFlowPyrLK(const cv::Mat& src1, const cv::Mat& src2, mvo::Feature& next)
{
    next.mfeatures.clear(); next.mstatus.clear(); next.merr.clear();
    cv::Size winSize = cv::Size(21,21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.11);
    
    cv::calcOpticalFlowPyrLK(src1, src2, mfeatures, next.mfeatures, next.mstatus, next.merr,
                             winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for(int i = 0; i < next.mstatus.size(); i++)
    {
        cv::Point2f pt = next.mfeatures.at(i - indexCorrection);

        if((next.mstatus.at(i) == 0) || 
            (pt.x < 0 || pt.x > (float)1241.0) || 
            (pt.y < 0 || pt.y > (float)376.0) ||
            next.merr.at(i) > ERRORSIZE)
        {
            if((pt.x < 0 || pt.x > (float)1241.0) || 
                (pt.y < 0 || pt.y > (float)376.0) ||
                next.merr.at(i) > ERRORSIZE)
            {
                next.mstatus.at(i) = 0;
            }
            mfeatures.erase(mfeatures.begin() + (i - indexCorrection));           // time complexity
            next.mfeatures.erase(next.mfeatures.begin() + (i - indexCorrection));
            mvdesc.erase(mvdesc.begin() + (i - indexCorrection));           // time complexity
            next.mvdesc.erase(next.mvdesc.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    
    if(next.mfeatures.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK" << std::endl;
        return false;
    }
    if(next.mvdesc.empty())
    {
        std::cerr << "failed calcOpticalFlowPyrLK descriptor" << std::endl;
        return false;
    }
    return true;
}

void ManageTrackPoints(const mvo::Feature& present, mvo::Feature& before)
{
    int indexCorrection = 0;
    for(int i = 0; i < present.mstatus.size(); i++)
    {
        if(present.mstatus.at(i) == 0)
        {
            before.mfeatures.erase(before.mfeatures.begin() + (i-indexCorrection));
            before.mvdesc.erase(before.mvdesc.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}
 
bool KeyPointToVec(const std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& features2d)
{
    cv::Point2f temp;
    for(int i=0; i<kp.size(); i++)
    {
        temp.x = kp.at(i).pt.x;
        temp.y = kp.at(i).pt.y;
        features2d.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool VecToKeyPoint(const std::vector<cv::Point2f>& features2d, std::vector<cv::KeyPoint>& kp)
{
    cv::KeyPoint temp;
    for(int i=0; i<features2d.size(); i++)
    {
        temp.pt.x = features2d.at(i).x;
        temp.pt.y = features2d.at(i).y;
        kp.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool MatToVec(const cv::Mat m, std::vector<std::vector<DTYPE>>& v)
{
    std::vector<DTYPE> temp;
    for(int j=0; j<m.rows ; j++)
    {
        for(int i=0; i<m.cols; i++)
        {
            temp.emplace_back(m.at<uchar>(j, i));
        }
        v.emplace_back(std::move(temp));
        temp.clear();
    }

    if(m.rows != v.size()) return false;

    return true;
}