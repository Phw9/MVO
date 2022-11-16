#include "Triangulate.h"
#include "Feature.h"
mvo::Triangulate::Triangulate(): mworldMapPoints{cv::Mat()}
{
    mworldMapPointsV.clear();
};

// bool mvo::Triangulate::CalcWorldPoints(const cv::Mat& pose1,
//                                         const cv::Mat& pose2,
//                                         const std::vector<cv::Point2f>& pts1,
//                                         const std::vector<cv::Point2f>& pts2)
// {
//     cv::triangulatePoints(pose1, pose2, pts1, pts2, mworldMapPoints);

//     std::cout << "2d point nums: " << pts1.size() << std::endl;
//     std::cout << "3d point nums: " << mworldMapPoints.size() << std::endl;
//     std::cout << "mworldMapPoints rows: " << mworldMapPoints.rows << std::endl;
//     std::cout << "mworldMapPoints cols: " << mworldMapPoints.cols << std::endl;
//     if(mworldMapPoints.empty())
//     {
//         std::cerr << "failed triagulatePoints" << std::endl;
//         return false;
//     }

//     return true;
// }
bool mvo::Triangulate::CalcWorldPoints(const cv::Mat& pose1,
                                        const cv::Mat& pose2,
                                        const mvo::Feature& pts1,
                                        const mvo::Feature& pts2)
{
    cv::triangulatePoints(pose1, pose2, pts1.mfeatures, pts2.mfeatures, mworldMapPoints);

    std::cout << "2d point nums: " << pts1.mfeatures.size() << std::endl;
    std::cout << "3d point nums: " << mworldMapPoints.size() << std::endl;
    std::cout << "mworldMapPoints rows: " << mworldMapPoints.rows << std::endl;
    std::cout << "mworldMapPoints cols: " << mworldMapPoints.cols << std::endl;
    if(mworldMapPoints.empty())
    {
        std::cerr << "failed triagulatePoints" << std::endl;
        return false;
    }

    return true;
}

bool mvo::Triangulate::ScalingPoints()
{
    for(int i = 0; i < mworldMapPoints.cols; i++)
    {
        for(int j = 0; j < mworldMapPoints.rows; j++)
        {
            mworldMapPoints.at<float>(j,i) = mworldMapPoints.at<float>(j,i) / mworldMapPoints.at<float>(mworldMapPoints.rows-1,i);
        }
    }
    if(mworldMapPoints.at<float>(mworldMapPoints.rows-1,0) != 1.0f) return false;
    return true;
}

void mvo::Triangulate::MatToPoints3f()
{
    cv::Point3f temp;
    for (int i = 0; i < mworldMapPoints.cols; i++) 
    {
		temp.x = mworldMapPoints.at<float>(0, i);
        temp.y = mworldMapPoints.at<float>(1, i);
        temp.z = mworldMapPoints.at<float>(2, i);
        mworldMapPointsV.emplace_back(std::move(temp));
        // std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;
	}
}


bool ManageMapPoints(const std::vector<uchar>& mstatus, std::vector<cv::Point3f>& map)
{
    int indexCorrection = 0;
    
    if(map.size() == mstatus.size())
    {
        for(int i = 0; i < mstatus.size(); i++)
        {
            if(mstatus.at(i) == 0)
            {
                map.erase(map.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
        // for(int i = 0; i < map.size(); i++)
        // {
        //     std::cout << map.at(i).x << " " << map.at(i).y << " " << map.at(i).z << std::endl;
        // }
        return true;
    }
    return false;
}

bool ManageMinusLocal(std::vector<mvo::Feature>& localTrackPoints, const std::vector<int>& id)
{
    for(int i = 0; i<localTrackPoints.size(); i++)
    {
        for(int j = 0; j<id.size(); j++)
        {
            localTrackPoints.at(i).mfeatures.erase(localTrackPoints.at(i).mfeatures.begin()+id.at(j));
        }
    }
    
    if(localTrackPoints.at(0).mfeatures.size() == localTrackPoints.at(localTrackPoints.size()-1).mfeatures.size()) return true;
    
    return false;
}

bool ManageMinusZ(mvo::Triangulate& map, cv::Mat& R, std::vector<int>& id)
{
    double tempd = 0;
    for(int i = 0; i < map.mworldMapPointsV.size(); i++)
    {
        tempd = (R.at<double>(2,0)*map.mworldMapPointsV.at(i).x) +
                (R.at<double>(2,1)*map.mworldMapPointsV.at(i).y) +
                (R.at<double>(2,2)*map.mworldMapPointsV.at(i).z);
        if(tempd < 0)
        {
            id.emplace_back(i);
            map.mworldMapPointsV.erase(map.mworldMapPointsV.begin()+i);
        }
    }
    if(map.mworldMapPointsV.size() == 0) return false;

    return true;
}

void ManageInlier(std::vector<mvo::Feature>& features2d, std::vector<cv::Point3f>& mapPoints3d, const cv::Mat& inlier)
{
    int interval;
    std::vector<int> idv;
    int ilast = inlier.at<int>(inlier.rows-1,0);
    int flast = features2d.at(0).mfeatures.size() - 1;
    idv.reserve(3000);
    idv.emplace_back(inlier.at<int>(0,0)-1);
    for(int i = 0; i < inlier.rows; i++)
    {
        idv.emplace_back(inlier.at<int>(i,0));
    }

    for(int i=0; i<idv.size()-1; i++)
    {
        interval = idv.at(i+1) - idv.at(i)-1;
        for(int j = i; j<interval+i; j++)
        {
            for(int k = 0; k < features2d.size(); k++)
            {
                features2d.at(k).mfeatures.erase(features2d.at(k).mfeatures.begin()+j);
            }
            mapPoints3d.erase(mapPoints3d.begin()+j);  
        }

        if(i == idv.size()-2 && flast-ilast != 0)
        {
            for(int l = 0; l < features2d.size(); l++)
            {
                for(int h = 0; h < flast-ilast; h++)
                {
                    features2d.at(l).mfeatures.pop_back();
                    features2d.at(l).mfeatures.pop_back();
                }
            }
            for(int h = 0; h < flast-ilast; h++)
            {
                mapPoints3d.pop_back();
                mapPoints3d.pop_back();
            }
            break;
        }
    }
}