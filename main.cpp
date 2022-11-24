#include "Config.h"
#include "Feature.h"
#include "Init.h"
#include "PoseEstimation.h"
#include "Triangulate.h"
#include "MapData.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"



int main(int argc, char** argv)
{
    std::ofstream rawData ("./image.txt", rawData.out | rawData.trunc);
	std::ifstream read ("./image.txt", read.in);
	std::ifstream readGTtvec ("../image/GTpose.txt", readGTtvec.in);
    
	std::vector<cv::Vec3f> readtvecOfGT;
	std::vector<cv::Vec3f> tvecOfGT;
	if(!read.is_open())
	{
		std::cerr << "file can't read image" << std::endl;
		return 0;
	}
	std::deque<std::string> readImageName;
	cv::Mat img;
	int imageCurNum = 0;
	int realFrame = 0;
	float RH = 100; float SH, SF;

	MakeTextFile(rawData, IMAGENUM);
	FileRead(readImageName, read);
	GTPoseRead(readtvecOfGT, readGTtvec);
	

	mvo::Feature detector;
	mvo::Feature trackerA, trackerB;
	std::vector<mvo::Feature> localTrackPointsA;
	localTrackPointsA.reserve(300);
	int lTPA = 0;
	std::vector<mvo::Feature> localTrackPointsB;
	localTrackPointsB.reserve(300);
	int lTPB = 0;
	std::vector<uchar> stats;

	mvo::StrctureFromMotion getEssential;
	mvo::PoseEstimation getPose;
	mvo::Initializer checkScore;

	std::vector<int> mapStats;
	mvo::Triangulate mapPointsA, mapPointsB;
	std::vector<cv::Point3f> localMapPointsA, localMapPointsB;
	std::vector<mvo::Triangulate> globalLandMark;
	int gLM = 0;
	int gKF = 0;

	float inlierRatio = 1000.0f;
	double angularVelocity = 0;
	std::vector<cv::Mat> globalRTMat; std::vector<cv::Mat> globalRMat;
	std::vector<cv::Vec3d> globalRVec; std::vector<cv::Vec3d> globalTVec;
	mvo::MapData setMapData;
	std::vector<mvo::MapData> globalMapData;
	globalMapData.reserve(1000);
	int gP = 0;
	int gD = 0;
	
	Viewer::MyVisualize pangolinViewer=Viewer::MyVisualize(WINDOWWIDTH, WINDOWHEIGHT);
    pangolinViewer.initialize();
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(WINDOWWIDTH, WINDOWHEIGHT, VIEWPOINTF, VIEWPOINTF, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(VIEWPOINTX, VIEWPOINTY, VIEWPOINTZ, 0, 0, 0, 0.0, -1.0, 0.0));
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -pangolinViewer.window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam));
	
	while(true)
	{
		if(initialNum == 1)
		{
			// Automatically Initilizer Calculate RH = SH/(SH+SF)
			if(realFrame == 0)	// feature extract
			{
				img = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				if (img.empty())
				{
					std::cerr << "frame upload failed" << std::endl;
				}				
				if(!trackerA.GoodFeaturesToTrack(img))
				{	
					std::cerr << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA));
				tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
				imageCurNum++;
				realFrame++;
			}
			while(initialNum!=0)
			{
				img = cv::imread(readImageName.at(imageCurNum), 
									cv::ImreadModes::IMREAD_UNCHANGED);
				if (img.empty())
				{
					std::cerr << "frame upload failed" << std::endl;
				}
				localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
															cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
				localTrackPointsA.emplace_back(std::move(trackerA));
				lTPA++;

				std::cout << "lTPA: " << lTPA << std::endl;

				for(int i = 0; i < lTPA-1; i++)
				{
					ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
					// std::cout << "size" << i <<": " << localTrackPointsA.at(i).mfeatures.size() << " ";
					// std::cout << "descsize" << i <<": " << localTrackPointsA.at(i).mvdesc.size() << std::endl;
				}

				SF = checkScore.CheckFundamental(localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures, 100.0f);
				SH = checkScore.CheckHomography(localTrackPointsA.at(0).mfeatures, localTrackPointsA.at(lTPA).mfeatures, 200.0f);
				RH = SH/(SH+SF);
				// std::cout << "score: " << RH << std::endl;
				
				if(RH>0.45 || SH<500.0f)
				{
					getEssential.CreateEssentialMatrix(localTrackPointsA.at(0).mfeatures, localTrackPointsA[lTPA].mfeatures, intrinsicKf);
					std::cout << "RH>0.45" << std::endl;
					globalRTMat.emplace_back(getEssential.mCombineRt); globalRMat.emplace_back(getEssential.mRotation);
					globalRVec.emplace_back(getEssential.mrvec); globalTVec.emplace_back(getEssential.mTranslation);
					gP++;
					setMapData.GetSFMPose(getEssential);
					if(!mapPointsA.CalcWorldPoints(m1, setMapData.mglobalRTMat,
							localTrackPointsA[0], localTrackPointsA[lTPA]))
					{
						std::cerr << "fail calc" << std::endl;
					}
					if(!ManageMinusZ(mapPointsA, setMapData.mglobalRMat, mapStats))
					{
						std::cerr << "failed ManageMinusZ A" << std::endl;
					}
					if(!ManageMinusLocal(localTrackPointsA, mapStats))
					{
						std::cerr << "failed Minus local" << std::endl;
					}

					setMapData.Get2DPoints(localTrackPointsA.at(lTPA));
					setMapData.Get3DPoints(mapPointsA);
					globalMapData.emplace_back(std::move(setMapData));

					std::cout << "statssize: " << localTrackPointsA.at(lTPA).mstatus.size()<< std::endl;
					std::cout << "mapPointsA.mworldMapPointsV: " << mapPointsA.mworldMapPointsV.size() << std::endl;
					std::cout << "localTrackPoints 0: " << localTrackPointsA.at(0).mfeatures.size() << std::endl;
					std::cout << "localtrackPoints l: " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;

					
					if(!trackerB.GoodFeaturesToTrack(img))
					{	
						std::cout << "new tracker B" << std::endl;
					}
					localTrackPointsB.emplace_back(std::move(trackerB));

					initialNum = 0;
				}
				cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				img = pangolinViewer.DrawFeatures(img, localTrackPointsA.at(lTPA-1).mfeatures, 
													localTrackPointsA.at(lTPA).mfeatures);
				tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
				std::cout << "KeyFrame(glboalMapData)size: " << globalMapData.size() << ", real:" << realFrame << ", image: " << imageCurNum <<std::endl;
				std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << std::endl;
				realFrame++;
				imageCurNum++;
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				d_cam.Activate(s_cam);
				// pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsA.mworldMapPointsV);
				pangolinViewer.DrawPoint(globalMapData, tvecOfGT);
				pangolin::FinishFrame();
				
				cv::imshow("img", img);
				if(cv::waitKey(0) == 27) break; // ESC key
			}
			std::cout << "=============  2-view end & start!!  =============" << std::endl;
		}
		// Start
		img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
		
		// if(keyframe gen){ generate Pose, MapPoints, new Track}

		// if num of Feature is less than NUMOFPOINTS, GFTT
		// while(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS)
		while((localTrackPointsA.size()> MINLOCAL && localTrackPointsB.size()> MINLOCAL) && 
				(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || 
				localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS || 
				angularVelocity > ANGULARVELOCITY ||
				(getPose.minlier.rows < NUMOFINLIER && gP>ESSENTIALFRAME) ||
				inlierRatio < INLIERRATIO))
		{
			imageCurNum--;
			img = cv::imread(readImageName.at(imageCurNum), 
						cv::ImreadModes::IMREAD_UNCHANGED);
			if(localTrackPointsA[lTPA].mfeatures.size() < NUMOFPOINTS || mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
			{
				// triangulate
				std::cout << "ㅇㅇㅇNewTrack Aㅇㅇㅇ" << std::endl;

				mapPointsB.CalcWorldPoints(globalMapData.at(gD).mglobalRTMat, getPose.mCombineRt, 
										localTrackPointsB.at(0), localTrackPointsB.at(lTPB));

				if(!ManageMinusZ(mapPointsB, getPose.mCombineRt, mapStats))
				{
					std::cerr << "failed ManageMinusZ B" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsB, mapStats))
				{
					std::cerr << "failed Minus local B" << std::endl;
				}
				globalLandMark.emplace_back(mapPointsB);
				gLM++;
 
				setMapData.GetPnPPose(getPose);
				setMapData.Get2DPoints(localTrackPointsB.at(lTPB));
				setMapData.Get3DPoints(mapPointsB);
				globalMapData.emplace_back(std::move(setMapData));
				// mvo::PushData(globalMapData, setMapData);
				gD++;

				localTrackPointsA.clear();
				if(!trackerA.GoodFeaturesToTrack(img))
				{
					std::cerr << "new tracker A" << std::endl;
				}
				localTrackPointsA.emplace_back(std::move(trackerA));
				lTPA = 0;
				std::cout << "localTrackPointsA size : " << localTrackPointsA.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(0).mfeatures.size() << std::endl;
				std::cout << "localTrackPointsB size : " << localTrackPointsB.size() << std::endl;
				std::cout << "localTrackPointsB.mfeature size : " << localTrackPointsB.at(lTPB).mfeatures.size() <<std::endl;
				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				gKF++;
				break;
			}

			if(localTrackPointsB[lTPB].mfeatures.size() < NUMOFPOINTS || mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
			{
				// triangulate
				std::cout << "ㅇㅇㅇNewTrack Bㅇㅇㅇ" << std::endl;

				mapPointsA.CalcWorldPoints(globalMapData.at(gD).mglobalRTMat, getPose.mCombineRt, 
										localTrackPointsA.at(0), localTrackPointsA.at(lTPA));

				if(!ManageMinusZ(mapPointsA, getPose.mCombineRt, mapStats))
				{
					std::cerr << "failed ManageMinusZ A" << std::endl;
				}
				if(!ManageMinusLocal(localTrackPointsA, mapStats))
				{
					std::cerr << "failed Minus local A" << std::endl;
				}
				globalLandMark.emplace_back(mapPointsA);
				gLM++;

				setMapData.GetPnPPose(getPose);
				setMapData.Get2DPoints(localTrackPointsA.at(lTPA));
				setMapData.Get3DPoints(mapPointsA);
				globalMapData.emplace_back(std::move(setMapData));
				// mvo::PushData(globalMapData, setMapData);
				gD++;

				localTrackPointsB.clear();
				if(!trackerB.GoodFeaturesToTrack(img))
				{	
					std::cout << "new tracker B" << std::endl;
				}
				localTrackPointsB.emplace_back(std::move(trackerB));
				lTPB = 0;
				std::cout << "localTrackPointsB size : " << localTrackPointsB.size() << std::endl;
				std::cout << "localTrackPointsB.mfeature size : " << localTrackPointsB.at(0).mfeatures.size() <<std::endl;
				std::cout << "localTrackPointsA size : " << localTrackPointsA.size() << std::endl;
				std::cout << "localTrackPointsA.mfeature size : " << localTrackPointsA.at(lTPA).mfeatures.size() << std::endl;
				imageCurNum++;
				img = cv::imread(readImageName.at(imageCurNum), 
							cv::ImreadModes::IMREAD_UNCHANGED);
				gKF++;
				break;
			}
			// imageCurNum++;
			// img = cv::imread(readImageName.at(imageCurNum), 
			// 			cv::ImreadModes::IMREAD_UNCHANGED);
		}

		// tracking A
		localTrackPointsA[lTPA].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerA);
		localTrackPointsA.emplace_back(std::move(trackerA));
		lTPA++;

		for(int i = 0; i < lTPA-1; i++)
		{
			ManageTrackPoints(localTrackPointsA.at(lTPA), localTrackPointsA.at(i));
		}
		// std::cout << "localTrackPointsA size: " << localTrackPointsA.size() << std::endl;

		if(!mapPointsA.ManageMapPoints(localTrackPointsA.at(lTPA).mstatus))
		{
			std::cerr << "failed manage MapPointsA" << std::endl;
		}
		
		if(mapPointsA.mworldMapPointsV.size() == localTrackPointsA.at(lTPA).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁTrackA SolvePnPㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ" << std::endl;

			getPose.solvePnP(mapPointsA.mworldMapPointsV, localTrackPointsA.at(lTPA).mfeatures, intrinsicKd);
			inlierRatio = (float)getPose.minlier.rows/(float)localTrackPointsA.at(lTPA).mfeatures.size();
			std::cout << "inlierRatio: " << inlierRatio << std::endl;
	
			
			globalRTMat.emplace_back(getPose.mCombineRt); globalRMat.emplace_back(getPose.mRotation);
			globalRVec.emplace_back(getPose.mrvec); globalTVec.emplace_back(getPose.mtranslation);
			gP++;
			
			angularVelocity = mvo::RotationAngle(globalRMat.at(gP-2), globalRMat.at(gP-1));

			std::cout << std::endl;
			std::cout << "mapPoints.size A: " << mapPointsA.mworldMapPointsV.size() << std::endl;
			std::cout << "before, present local points size A: " << localTrackPointsA[lTPA-1].mfeatures.size() <<
			", "<< localTrackPointsA[lTPA].mfeatures.size() << std::endl;
			std::cout << "mrvec: " << globalRVec.at(gP-1) << "  angularV: " << angularVelocity << std::endl;
			std::cout << "mtvec: " << globalTVec.at(gP-1) << std::endl;
		}
		std::cout << std::endl;
		// tacking B
		localTrackPointsB[lTPB].OpticalFlowPyrLK(cv::imread(readImageName.at(imageCurNum-1), 
													cv::ImreadModes::IMREAD_UNCHANGED), img, trackerB);
		localTrackPointsB.emplace_back(std::move(trackerB));
		lTPB++;
		for(int i = 0; i < lTPB-1; i++)
		{
			ManageTrackPoints(localTrackPointsB.at(lTPB), localTrackPointsB.at(i));
		}
		// std::cout << "lTPA: " << lTPA << std::endl;
		// std::cout << "lTPA feature size : " << localTrackPointsA[lTPA].mfeatures.size() << std::endl << std::endl;
		// std::cout << "statssize: " << localTrackPointsB.at(lTPB).mstatus.size() << std::endl;
		// std::cout << "before mapPointsB.mworldMapPointsV: " << mapPointsB.mworldMapPointsV.size() << std::endl;
		// std::cout << "before trackPoints: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;

		if(!mapPointsB.ManageMapPoints(localTrackPointsB.at(lTPB).mstatus))
		{
			std::cerr << "failed manage MapPointsB" << std::endl;
		}
		// std::cout << "after mapPointsB.mworldMapPointsV: " << mapPointsB.mworldMapPointsV.size() << std::endl;
		// std::cout << "after trackPoints: " << localTrackPointsB.at(lTPB).mfeatures.size() << std::endl;

		if(mapPointsB.mworldMapPointsV.size() == localTrackPointsB.at(lTPB).mfeatures.size())
		{
			std::cout << "ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁTrackB SolvePnPㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ" << std::endl;
			getPose.solvePnP(mapPointsB.mworldMapPointsV, localTrackPointsB.at(lTPB).mfeatures, intrinsicKf);
			inlierRatio = (float)getPose.minlier.rows/(float)localTrackPointsB.at(lTPB).mfeatures.size();
			std::cout << "inlierRatio: " << inlierRatio << std::endl;
	

			globalRTMat.emplace_back(getPose.mCombineRt); globalRMat.emplace_back(getPose.mRotation);
			globalRVec.emplace_back(getPose.mrvec); globalTVec.emplace_back(getPose.mtranslation);
			gP++;

			angularVelocity = mvo::RotationAngle(globalRMat.at(gP-2), globalRMat.at(gP-1));

			std::cout << std::endl;
			std::cout << "mapPoints.size B: " << mapPointsB.mworldMapPointsV.size() << std::endl;
			std::cout << "before, present local points size B: " << localTrackPointsB[lTPB-1].mfeatures.size() <<
			", "<< localTrackPointsB[lTPB].mfeatures.size() << std::endl;
			std::cout << "mrvec: " << globalRVec.at(gP-1) << "  angularV: " << angularVelocity << std::endl;
			std::cout << "mtvec: " << globalTVec.at(gP-1) << std::endl;
		}

		std::cout << std::endl;
		std::cout << "lTPA: " << lTPA << ", " << "lTPB: " << lTPB << std::endl;
		std::cout << "lTPA feature size : " << localTrackPointsA[lTPA].mfeatures.size() << std::endl;
		std::cout << "lTPB feature size : " << localTrackPointsB[lTPB].mfeatures.size() << std::endl;

		// draw tracking points
		if(localTrackPointsA[lTPA].mfeatures.size() > localTrackPointsB[lTPB].mfeatures.size())
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.DrawFeatures(img, localTrackPointsA.at(lTPA-1).mfeatures, 
											localTrackPointsA.at(lTPA).mfeatures);
		}
		else
		{
		cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		img = pangolinViewer.DrawFeatures(img, localTrackPointsB.at(lTPB-1).mfeatures, 
											localTrackPointsB.at(lTPB).mfeatures);		
		}
		tvecOfGT.emplace_back(readtvecOfGT.at(imageCurNum));
		std::cout << std::endl;
		std::cout << "KeyFrame(glboalMapData)size: " << globalMapData.size() << ", real:" << realFrame << ", image: " << imageCurNum <<std::endl;
		std::cout << "mpoint2D size: " << globalMapData.at(gD).mpoint2D.size() << std::endl;
		std::cout << "mpoint3D size: " << globalMapData.at(gD).mpoint3D.size() << std::endl;
		std::cout << "mvdesc size: " << globalMapData.at(gD).mvdesc.size() << std::endl;
		
		std::cout << "mtvec: " << globalTVec.at(gP-1) << " tvec size: " << globalTVec.size() << std::endl;
		std::cout << "gDtvec: " << globalMapData.at(gD).mglobaltvec << std::endl;
		std::cout << "tvecOfGT: " << tvecOfGT.at(imageCurNum) << "  gKF: " << gKF << " Img/KF: " << globalTVec.size()/(gD+1) << std::endl; 
		std::cout << "==============================================" << std::endl;
		imageCurNum++;
		realFrame++;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);
		// if(mapPointsA.mworldMapPointsV.size() > mapPointsB.mworldMapPointsV.size())
		// {
		// 	pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsA.mworldMapPointsV);
		// }
		// else
		// {
		// 	pangolinViewer.DrawPoint(globalTVec, tvecOfGT, globalLandMark, mapPointsB.mworldMapPointsV);
		// }
		pangolinViewer.DrawPoint(globalMapData, tvecOfGT);
    	pangolin::FinishFrame();
		cv::imshow("img", img);
		if(cv::waitKey(0) == 27) break; // ESC key
	}

    return 0;
}