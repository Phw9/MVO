
======================================================================================================
Monocular KLT based Visual Odometry
======================================================================================================
*/

// (0) Monocular visual Odometry 
#elif CAMERA_MODE==2
    printf("monocular visual odometry ====== ");
    sprintf(filename1, "../kitty/image_0/%06d.png", 0);
    sprintf(filename2, "../kitty/image_0/%06d.png", 1);

    //read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    Mat img_1, img_2;
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    vector<uchar> status;

    featureDetection(img_1, prevFeatures);

    vector<Point2f> keyframeFeatures = prevFeatures;
    vector<Point2f> keyFeatures_removed = prevFeatures;

    featureTracking(img_1, img_2, prevFeatures, currFeatures, status, keyFeatures_removed); //track those features to img_2


    //recovering the pose and the essential matrix
    Mat Kd;
    Mat R1 = Mat::eye(3, 3, CV_64FC1);
    A.convertTo(Kd, CV_64F);
    Mat E, R, t, mask;

    int id_explode=0;

    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Rt0 = Mat::eye(3, 4, CV_64FC1); //prev pose
    Rt1 = Mat::eye(3, 4, CV_64FC1); //next pose

    R.copyTo(Rt1.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Rt1.rowRange(0, 3).col(3));

    Mat final_Rt = Mat::eye(4, 4, CV_64FC1);

    Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));
    //cout << final_Rt.inv() << endl;

    Pango_REpose.push_back
    (cv::Point3f(final_Rt.at<double>(0,3), final_Rt.at<double>(1,3), final_Rt.at<double>(2,3)));

    R_f = R.clone();
    t_f = t.clone();

    keyframe_Rt2 = Rt0.clone();
    prevFeatures = currFeatures;

    Mat prevImage = img_2; // I(t)
    Mat currImage; // I(t+1)

    char filename[100];
    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
        for(int i=0; i<12; i++){
            is>>temp_number[i];
        }

        Pango_GTpose.push_back(cv::Point3f(temp_number[3],temp_number[7],temp_number[11]));

        sprintf(filename, "../kitty/image_0/%06d.png", numFrame);
        printf("\n\nsuccess to load ... kitty image : %d, mode : %d \n", numFrame, mode);
        Mat currImage_c = imread(filename);
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status, status1;

        if (mode == 1) {
            if (redetection_switch == true) {//2 :: Re-detection 
                //2-1 re-detection
                redetection_switch = false;
                prevFeatures = prevFeatures_2;
                featureDetection(prevImage, keyframeFeatures);
                keyFeatures_removed = keyframeFeatures;
                prevFeatures_2 = keyframeFeatures;
                keyframe_Rt2 = Rt1.clone();                
            }//2 :: step end 
            

            //3-1 : tracking (1), (2)  
            point3d_world = featureTracking_3(prevImage, currImage, prevFeatures, currFeatures, status, point3d_world, index_of_landmark); //number1 tracking line
            featureTracking(prevImage, currImage, prevFeatures_2, currFeatures_2, status1, keyFeatures_removed); //number2 tracking line
            Mat inlier;

            //3-2 : type-casting and Pose Estimation
            vector<Point2d> currFeatures_double;
            for(int i=0;i<currFeatures.size();i++){
                currFeatures_double.push_back(cv::Point2d(double(currFeatures[i].x),
                double(currFeatures[i].y)));
            }
            solvePnPRansac(point3d_world, currFeatures_double, A, noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE);

            double inlier_ratio = (double)inlier.rows / (double)point3d_world.rows;
            printf("3d point : %d,  inlier : %d\n", point3d_world.rows, inlier.rows);
            printf("inrier_Ratio : %f\n", inlier_ratio);

            //store inlier points 
            vector<cv::Point2d> tempFeatures;
            vector<cv::Point3d> tempFeatures1;

            vector<int> index_of_landmark_inlier; //inlier id 
            Store_inlier(inlier, currFeatures, point3d_world, tempFeatures, tempFeatures1, index_of_landmark, index_of_landmark_inlier);           
 
            //3-3 : Motion Estimation ( Pose Optimization )
            bundle::motion_only_BA(rvec, tvec, tempFeatures, tempFeatures1, focal, pp);
            cout<<"motion ba done"<<endl;
            UpdateRT(Rt0, Rt1, rvec, tvec, mode);

            Mat final_Rt = Mat::eye(4, 4, CV_64FC1);
            Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));
            Mat final_Rt_inv=final_Rt.inv();
            final_Rt_inv.pop_back();

            int rotate_switch=false;
            rotate_switch=Keypoint::Calculate_Angle(Rt1, key_RT);

            //4 : New keyframe selection 
            if (inlier.rows<150||inlier_ratio<0.4||rotate_switch==true) {
                printf("=================redetection is conducted=================\n");
                redetection_switch = true; 
                draw_=true;

                //4-1) Triangulation between keyframe and current frame of re-detection line
                // localBA (0) : Triangulation
                vector<cv::Point2d> triangulation_points1, triangulation_points2;
                for (int i = 0; i < keyFeatures_removed.size(); i++) {
                    triangulation_points1.push_back
                        (cv::Point2d(keyFeatures_removed[i].x,keyFeatures_removed[i].y));
                    triangulation_points2.push_back
                        (cv::Point2d(currFeatures_2[i].x, currFeatures_2[i].y));
                }

                triangulatePoints(Kd * keyframe_Rt2, Kd * Rt1, triangulation_points1, triangulation_points2, point3d_homo);
                point3d_world = convert_14to13(point3d_homo);


                // localBA (1) : Erase outliers
                printf(" point3d : %d\n", point3d_world.rows);
                
                // localBA (2) : id-generation
                for(int i=0;i<currFeatures_2.size();i++){
                    index_of_landmark_2.push_back(count_keypoints+i);
                }

                Keypoint::Erase_BackPoints(currFeatures_2, point3d_world, final_Rt_inv, index_of_landmark_2);
                // Keypoint::Erase_Outliers(currFeatures_2, point3d_world, Rt1, Kd, index_of_landmark_2);

                // localBA (3) : id-matching with inliers 
                for (int i=0;i<tempFeatures.size();i++){
                    bool match=false;
                    float x2=float(tempFeatures[i].x);
                    float y2=float(tempFeatures[i].y);
                    int temp_id=index_of_landmark_inlier[i];

                    for(int j=0;j<currFeatures_2.size();j++){
                        float x1=float(currFeatures_2[j].x);
                        float y1=float(currFeatures_2[j].y);
                        
                        float distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
                        if(distance<distance_num){
                            index_of_landmark_2[j]=temp_id;

                            currFeatures_2[j].x=tempFeatures[i].x;
                            currFeatures_2[j].y=tempFeatures[i].y;

                            point3d_world.at<double>(j,0)=tempFeatures1[i].x;
                            point3d_world.at<double>(j,1)=tempFeatures1[i].y;
                            point3d_world.at<double>(j,2)=tempFeatures1[i].z;

                            match=true;
                            break;
                        }
                    }
                }


                int temp_count_keypoints=count_keypoints+index_of_landmark_2.size()+1000;


                vector<int> temp_index_of_landmark_2=index_of_landmark_2;
                vector<Point2f> temp_currFeatures_2_=currFeatures_2;
                Mat temp_point3d_world=point3d_world.clone();

                // Keypoint::Remain_Inlier(temp_currFeatures_2_, temp_point3d_world, Rt1, temp_index_of_landmark_2, count_keypoints);
                Keypoint::Erase_Outliers(temp_currFeatures_2_, temp_point3d_world, Rt1, Kd, temp_index_of_landmark_2);
                printf("remain inlier point3d size : %d \n",temp_point3d_world.rows);

                vector<Point2d> temp_currFeatures_2;
                for(int i=0;i<currFeatures_2.size();i++){ 
                    temp_currFeatures_2.push_back(
                        cv::Point2d(double(temp_currFeatures_2_[i].x), double(temp_currFeatures_2_[i].y)));
                }               

                //type-casting point2f to point2d  ;
                vector<Point2d> currFeatures_2_double;
                for(int i=0;i<currFeatures_2.size();i++){ 
                    currFeatures_2_double.push_back(
                        cv::Point2d(currFeatures_2[i].x,currFeatures_2[i].y));
                }
                // insert information of current keyframe (index, features) 
                land_mark.InsertMappoint_2(point3d_world, currFeatures_2_double, KeyframeNum, index_of_landmark_2);
                land_mark.push_each_keyframe_information(currFeatures_2_double, KeyframeNum, index_of_landmark_2);

                map<int,Point3d> mappoint=land_mark.getMappoint();
                for(int i=0;i<point3d_world.rows;i++){
                    int temp_id=index_of_landmark_2[i];
                        point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                        point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                        point3d_world.at<double>(i,2)=mappoint[temp_id].z;
                }

                index_of_landmark=index_of_landmark_2;
                land_mark.push_inlier_list(temp_currFeatures_2, KeyframeNum, temp_index_of_landmark_2);

                //all
                for(int i=0;i<index_of_landmark.size();i++){
                    local_BA_id_list.push_back(index_of_landmark[i]);
                }
                local_BA_points_size_list.push_back(point3d_world.rows);

                for(int i=0;i<index_of_landmark.size();i++){
                    local_BA_id_list_inliers.push_back(temp_index_of_landmark_2[i]);
                }
                local_BA_points_size_list_inliers.push_back(temp_point3d_world.rows);



                // 3-5) Local BA start
                //insert pose
                if(land_mark.Getrvec_size()<local_ba_frame){
                    land_mark.InsertPose(rvec,tvec);
                }
                else{
                    land_mark.pop_Pose();
                    land_mark.InsertPose(rvec,tvec);
                }
 

                //erase overlab all points
                vector<int> temp_local_BA_id_list=local_BA_id_list;
                sort(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end());
                temp_local_BA_id_list.erase(unique(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end()),temp_local_BA_id_list.end());

                vector<int> temp_local_BA_id_list_inliers=local_BA_id_list_inliers;
                sort(temp_local_BA_id_list_inliers.begin(),temp_local_BA_id_list_inliers.end());
                temp_local_BA_id_list_inliers.erase(unique(temp_local_BA_id_list_inliers.begin(),temp_local_BA_id_list_inliers.end()),temp_local_BA_id_list_inliers.end());

                bundle::Local_BA(land_mark, focal, pp, rvec, tvec, point3d_world, KeyframeNum, index_of_landmark_2, temp_local_BA_id_list_inliers ,1);
                UpdateRT(Rt0, Rt1, rvec, tvec, mode);

                if(land_mark.Getrvec_size()==local_ba_frame){
                    //all points
                    int zahnm=local_BA_points_size_list.front();
                    for(int i=0;i<zahnm;i++){
                        local_BA_id_list.erase(local_BA_id_list.begin());
                    }
                    local_BA_points_size_list.erase(local_BA_points_size_list.begin());
                }

                if(land_mark.Getrvec_size()==local_ba_frame){
                    //all points
                    int zahnm=local_BA_points_size_list_inliers.front();
                    for(int i=0;i<zahnm;i++){
                        local_BA_id_list_inliers.erase(local_BA_id_list_inliers.begin());
                    }
                    local_BA_points_size_list_inliers.erase(local_BA_points_size_list_inliers.begin());
                }
                key_RT=Rt1.clone();
                KeyframeNum++;
                index_of_landmark_2.clear();
                count_keypoints=temp_count_keypoints;
            }
            Pango_Map=point3d_world.clone();

            /* visualization prevFeatures and triangulated Features */
            prevFeatures_2 = currFeatures_2;
            final_Rt = Mat::eye(4, 4, CV_64FC1);
            Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));

            final_Rt_inv=final_Rt.inv();
            final_Rt_inv.pop_back();

            Pango_REpose.push_back
            (cv::Point3f(final_Rt_inv.at<double>(0,3), final_Rt_inv.at<double>(1,3), final_Rt_inv.at<double>(2,3)));
            Mat prevImage_d = prevImage.clone();
            cvtColor(prevImage_d, prevImage_d, COLOR_GRAY2BGR);


            if(draw_==false){
                prevImage_d=pangolin_viewer.cv_draw_features(prevImage_d,point3d_world, Kd, Rt1, currFeatures);
            }
            else{
                draw_=false;
                prevImage_d=pangolin_viewer.cv_draw_features(prevImage_d,point3d_world, Kd, Rt1, currFeatures_2);
            }
            imshow("CV viewer", prevImage_d);

            if (numFrame >= 6) {
                waitKey(0);
            }
        }


        // Structure From Motion ( From first frame to 5-frame )====================================================
        else if (mode == 0) { 
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, keyFeatures_removed);
            E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
            recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

            UpdateRT(Rt0, Rt1, R, t, mode);

            t_f = t_f + (R_f * t);
            R_f = R * R_f;

            Rt_f = Mat::eye(3, 4, CV_64FC1);
            R_f.copyTo(Rt_f.rowRange(0, 3).colRange(0, 3));
            t_f.copyTo(Rt_f.rowRange(0, 3).col(3));

            Pango_REpose.push_back // for Pose visualization
            (cv::Point3f(Rt_f.at<double>(0,3), Rt_f.at<double>(1,3), Rt_f.at<double>(2,3)));

            if (numFrame == 10) {
                mode = 1;
                redetection_switch = true;
                vector<Point2f> currFeatures_temp;
                vector<Point2f> keyFeatures_removed_temp;

                vector<cv::Point2d> triangulation_points1, triangulation_points2;


                for (int i = 0; i < keyFeatures_removed.size(); i++) {
                    if(mask.at<bool>(i,0)==1){
                        // Point2d type : only triangulation
                        triangulation_points1.push_back
                            (cv::Point2d((double)keyFeatures_removed[i].x, (double)keyFeatures_removed[i].y));
                        triangulation_points2.push_back
                            (cv::Point2d((double)currFeatures[i].x, (double)currFeatures[i].y));

                        // Point2f type : only remaining inliers 
                        currFeatures_temp.push_back
                            (cv::Point2f(currFeatures[i].x,currFeatures[i].y));
                        keyFeatures_removed_temp.push_back
                            (cv::Point2f(keyFeatures_removed[i].x,keyFeatures_removed[i].y));                            
                    }
                }
                /*============================ Triangulation step =================================*/
                Mat Rt_f44 = Mat::eye(4, 4, CV_64FC1);
                Mat Rt_f44_inv;
                Rt_f.copyTo(Rt_f44.rowRange(0, 3).colRange(0, 4));        

                Rt_f44_inv = Rt_f44.inv();
                Mat tepMat1 = Rt_f44_inv.clone();
                tepMat1.pop_back();

                triangulatePoints(Kd * keyframe_Rt2, Kd * tepMat1, triangulation_points1, triangulation_points2, point3d_homo);
                point3d_world = convert_14to13(point3d_homo);

                Keypoint::Erase_BackPoints(currFeatures_temp, keyFeatures_removed_temp, point3d_world, Rt_f);


                currFeatures=currFeatures_temp;
                keyFeatures_removed=keyFeatures_removed_temp;

                //keyframe 1 
                for(int i=0;i<point3d_world.rows;i++){
                    index_of_landmark.push_back(i);
                }

                vector<Point2d> keyFeatures_removed_double;
                vector<Point2d> currFeatures_double;

                for(int i=0;i<currFeatures.size();i++){
                    keyFeatures_removed_double.push_back(
                        cv::Point2d(double(keyFeatures_removed[i].x),double(keyFeatures_removed[i].y)));
                    currFeatures_double.push_back(
                        cv::Point2d(double(currFeatures[i].x),double(currFeatures[i].y)));
                }

                //all points
                land_mark.push_each_keyframe_information(keyFeatures_removed_double, 0, index_of_landmark);
                land_mark.push_each_keyframe_information(currFeatures_double, 1, index_of_landmark);
                land_mark.InsertMappoint_2(point3d_world, currFeatures_double, 1, index_of_landmark);

                local_BA_id_list=index_of_landmark;

                local_BA_points_size_list.push_back(point3d_world.rows);
                local_BA_points_size_list.push_back(point3d_world.rows);

                //inlier points
                land_mark.push_inlier_list(keyFeatures_removed_double, 0, index_of_landmark);
                land_mark.push_inlier_list(currFeatures_double, 1, index_of_landmark);
                local_BA_id_list_inliers=index_of_landmark;

                local_BA_points_size_list_inliers.push_back(point3d_world.rows);
                local_BA_points_size_list_inliers.push_back(point3d_world.rows);

                //keyfrmae 0,1 Pose Insert
                Mat tmp_tvec=Mat::eye(3,1,CV_64FC1);
                Mat tmp_rvec;
                Mat Rt0_tmp=Mat::eye(3,3,CV_64FC1);

                keyframe_Rt2.rowRange(0,3).colRange(0,3).copyTo(Rt0_tmp.rowRange(0,3).colRange(0,3));
                keyframe_Rt2.rowRange(0,3).col(3).copyTo(tmp_tvec);
                Rodrigues(Rt0_tmp, tmp_rvec);

                Mat tmp_tvec1=Mat::eye(3,1,CV_64FC1);
                Mat tmp_rvec1;
                Mat Rt1_tmp=Mat::eye(3,3,CV_64FC1);

                tepMat1.rowRange(0,3).colRange(0,3).copyTo(Rt1_tmp.rowRange(0,3).colRange(0,3));
                tepMat1.rowRange(0,3).col(3).copyTo(tmp_tvec1);
                Rodrigues(Rt1_tmp,tmp_rvec1);

                land_mark.InsertPose(tmp_rvec,tmp_tvec);
                land_mark.InsertPose(tmp_rvec1,tmp_tvec1);

                prevFeatures_2 = currFeatures;
                Rt1 = tepMat1.clone();
                key_RT=tepMat1.clone();
                
                KeyframeNum=2;
                count_keypoints=point3d_world.rows*2+1000;
            }
        }
        if(numFrame<10){
            cout<<Rt_f<<endl;
        }
        else{
            cout<<Rt1<<endl;
        }

        waitKey(1);

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        /*pangolin code*/
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        vector<Point3d> hello=land_mark.Getvector();
        pangolin_viewer.draw_point(Pango_REpose, Pango_GTpose, hello, Pango_Map);
        pangolin::FinishFrame();
    }
#endif 
    return 0;
}