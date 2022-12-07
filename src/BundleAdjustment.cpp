#include "BundleAdjustment.h"

mvo::SnavelyReprojectionError::SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d worldHomoGen4d, double focal, double ppx, double ppy)
            : observed_x{observed_x}, observed_y{observed_y}, worldHomoGen4d{worldHomoGen4d}, focal{focald}, ppx{camX}, ppy{camY} {}

template <typename T> bool mvo::SnavelyReprojectionError::operator()(const T *const rvec_eig,
                        const T *const tvec_eig,
                        T *residuals) const
{

    const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);

    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    Eigen::Matrix<T, 3, 4> worldToCam;
    worldToCam << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 3, 1> pixel;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;


    pixel = Kd.cast<T>() * worldToCam * worldHomoGen4d.cast<T>();

    T predicted_x = (pixel[0] / pixel[2]);
    T predicted_y = (pixel[1] / pixel[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
}

mvo::SnavelyReprojectionErrorLocal::SnavelyReprojectionErrorLocal(double observed_x, double observed_y,
                                                                  double focal, double ppx, double ppy)
                    : observed_x{observed_x}, observed_y{observed_y}, focal{focald}, ppx{camX}, ppy{camY} {}

template <typename T> bool mvo::SnavelyReprojectionErrorLocal::operator()(const T *const rvec_eig,
                                              const T *const tvec_eig,
                                              const T *const worldPoint,
                                              T *residuals) const
{
    const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);

    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    Eigen::Matrix<T, 3, 4> worldToCam;
    worldToCam << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 3, 1> pixel;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    Eigen::Matrix<T, 4, 1> worldPoint_eig(worldPoint[0], worldPoint[1], worldPoint[2], static_cast<T>(1));

    pixel = Kd.cast<T>() * worldToCam * worldPoint_eig;

    T predicted_x = (pixel[0] / pixel[2]);
    T predicted_y = (pixel[1] / pixel[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
}                                            


mvo::BundleAdjustment::BundleAdjustment(mvo::Feature& ft, mvo::PoseEstimation& rt, mvo::Triangulate& tri)
                    : ft{ft}, rt{rt}, tri{tri}{}

mvo::BundleAdjustment::BundleAdjustment(mvo::MapData& data): data{data}{}
    

bool mvo::BundleAdjustment::MotionOnlyBA()
{
    Eigen::Vector3d rvec_eig;
    Eigen::Vector3d tvec_eig;
    int N = ft.mfeatures.size();
    rvec_eig[0]=rt.mrvec[0]; rvec_eig[1]=rt.mrvec[1]; rvec_eig[2]=rt.mrvec[2];
    tvec_eig[0]=rt.mtvec.at<double>(0); tvec_eig[1]=rt.mtvec.at<double>(1); tvec_eig[2]=rt.mtvec.at<double>(2);

    Eigen::MatrixXd points2d_eig(2, N);
    for(int i = 0; i < N; i++)
    {
        points2d_eig(0,i)=ft.mfeatures.at(i).x;
        points2d_eig(1,i)=ft.mfeatures.at(i).y;
    }
    
    Eigen::MatrixXd points3d_eig(4, tri.mworldMapPointsV.size());

    if(ft.mfeatures.size() != tri.mworldMapPointsV.size())
    {
        std::cerr << "Don't correct matching in Motion only BA" << std::endl;
        return false;
    }
    int M = tri.mworldMapPointsV.size();
    for(int i = 0; i < M; i++)
    {
        points3d_eig(0,i)=tri.mworldMapPointsV.at(i).x;
        points3d_eig(1,i)=tri.mworldMapPointsV.at(i).y;
        points3d_eig(2,i)=tri.mworldMapPointsV.at(i).z;
        points3d_eig(3,i)=1;
    }

    ceres::Problem problem;

    for(int i = 0; i < M; i++)
    {
        ceres::CostFunction* cost_function=
            new ceres::AutoDiffCostFunction<mvo::SnavelyReprojectionError, 2, 3, 3>
            (new mvo::SnavelyReprojectionError(points2d_eig(0,i), points2d_eig(1,i), points3d_eig.col(i), focald, camX, camY));
        ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(cost_function, loss, rvec_eig.data(), tvec_eig.data());
    }

    ceres::Solver::Options options;
    
    options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations=100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;

    for(int i=0; i<3; i++)
    {
        rt.mrvec = double(rvec_eig[i]);
        rt.mtvec.at<double>(i) = double(tvec_eig[i]);
    }
    return true;
}

bool mvo::BundleAdjustment::LocalBA(int gD, std::vector<mvo::MapData>& map, mvo::Covisibilgraph& cov)
{
    Eigen::MatrixXd rvec_local(3, gD+1);
    Eigen::MatrixXd tvec_local(3, gD+1);
    std::vector<Eigen::MatrixXd> points2d;
    std::vector<Eigen::MatrixXd> points3d;
    ceres::Problem problem;

    for(int i=0; i<gD; i++)
    {
        rvec_local(0,i) = map.at(i).mglobalrvec[0];
        rvec_local(1,i) = map.at(i).mglobalrvec[1];
        rvec_local(2,i) = map.at(i).mglobalrvec[2];

        tvec_local(0,i) = map.at(i).mglobaltvec[0];
        tvec_local(1,i) = map.at(i).mglobaltvec[1];
        tvec_local(2,i) = map.at(i).mglobaltvec[2];
        
        if(map.at(i).mpoint2D.size() != map.at(i).mpoint3D.size())
        {
            std::cerr<< "localBA Matching error" << std::endl;
            return false;
        }

        int N = map.at(i).mpoint2D.size();
        Eigen::MatrixXd points2d_eig(2, N);
        Eigen::MatrixXd points3d_eig(4, N);
            
        for(int j = 0; j < N; j++)
        {
            points2d_eig(0,j) = map.at(i).mpoint2D.at(j).x;
            points2d_eig(1,j) = map.at(i).mpoint2D.at(j).y;        
        }
        for(int j = 0; j < N; j++)
        {
            points3d_eig(0,i) = map.at(i).mpoint3D.at(j).x;
            points3d_eig(1,i) = map.at(i).mpoint3D.at(j).y;
            points3d_eig(2,i) = map.at(i).mpoint3D.at(j).z;
            points3d_eig(3,i) = 1;            
        }
        points2d.push_back(points2d_eig);
        points3d.emplace_back(std::move(points3d_eig));

        for(int j = 0; j < N; j++)
        {
            ceres::CostFunction* cost_function=
                new ceres::AutoDiffCostFunction<mvo::SnavelyReprojectionErrorLocal, 2, 3, 3, 3>
                (new mvo::SnavelyReprojectionErrorLocal(points2d_eig(0,j), points2d_eig(1,j), focald, camX, camY));
            ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);
            problem.AddResidualBlock(cost_function, loss, rvec_local.col(i).data(), tvec_local.col(i).data(), points3d.at(i).col(j).data());
        }
        

        if(i>0)
        {
            int M = cov.mgraph.at(i-1).size();
            for(int j=0; j<M; j++)
            {
                int prev = cov.mgraph.at(i-1).at(j).first;  //i-1 idx
                int next = cov.mgraph.at(i-1).at(j).second; // i idx
                std::cout << "j: " << j << ", prev: " << prev << ", next: " << next << std::endl;
                std::cout << "points3d.at(i).cols(): " << points3d.at(i).cols() << std::endl;
                ceres::CostFunction* cost_function1=
                new ceres::AutoDiffCostFunction<mvo::SnavelyReprojectionErrorLocal, 2, 3, 3, 3>
                (new mvo::SnavelyReprojectionErrorLocal(points2d.at(i-1)(0, prev), points2d.at(i-1)(1, prev), focald, camX, camY));
                ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);
                problem.AddResidualBlock(cost_function1, loss, rvec_local.col(i-1).data(), tvec_local.col(i-1).data(), points3d.at(i).col(next).data());
                // std::cout << "local-_-" << std::endl;
            }
        }
    }// AddResidualBlock end

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations=100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    
    // for(int i = 0; i<gD; i++)
    // {
    //     int L = map.at(i).mpoint3D.size();
    //     for(int j = 0; j < L; j++)
    //     {
    //         std::cout << "before(" << i << "," << j << "): " << map.at(i).mpoint3D.at(j).x << ", " << map.at(i).mpoint3D.at(j).y << ", " << map.at(i).mpoint3D.at(j).z << std::endl;
    //         std::cout << "after (" << i << "," << j << "): " << points3d.at(i)(0,j) << ", " << points3d.at(i)(1,j) << ", " << points3d.at(i)(2,j) << std::endl;
    //     }
    // }
    for(int i = 0; i<gD; i++)
    {
        map.at(i).mglobalrvec[0] = rvec_local(0,i);
        map.at(i).mglobalrvec[1] = rvec_local(1,i);
        map.at(i).mglobalrvec[2] = rvec_local(2,i);

        map.at(i).mglobaltvec[0] = tvec_local(0,i);
        map.at(i).mglobaltvec[1] = tvec_local(1,i);
        map.at(i).mglobaltvec[2] = tvec_local(2,i);

        int n = map.at(i).mpoint3D.size();
        for(int j = 0; j < n; j++)
        {
            map.at(i).mpoint3D.at(j).x = points3d.at(i)(0,j);
            map.at(i).mpoint3D.at(j).y = points3d.at(i)(1,j);
            map.at(i).mpoint3D.at(j).z = points3d.at(i)(2,j);
        }
    }

    return true;
}