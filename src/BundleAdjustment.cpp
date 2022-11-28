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

    Eigen::Matrix<T, 3, 1> pixel3d;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;

    pixel3d = Kd.cast<T>() * worldToCam * worldHomoGen4d.cast<T>();

    T predicted_x = (pixel3d[0] / pixel3d[2]);
    T predicted_y = (pixel3d[1] / pixel3d[2]);

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

    rvec_eig[0]=rt.mrvec[0]; rvec_eig[1]=rt.mrvec[1]; rvec_eig[2]=rt.mrvec[2];
    tvec_eig[0]=rt.mtvec.at<double>(0); tvec_eig[1]=rt.mtvec.at<double>(1); tvec_eig[2]=rt.mtvec.at<double>(2);

    Eigen::MatrixXd points2d_eig(2, ft.mfeatures.size());
        for(int i = 0; i < ft.mfeatures.size(); i++)
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

    for(int i = 0; i < tri.mworldMapPointsV.size(); i++)
    {
        points3d_eig(0,i)=tri.mworldMapPointsV.at(i).x;
        points3d_eig(1,i)=tri.mworldMapPointsV.at(i).y;
        points3d_eig(2,i)=tri.mworldMapPointsV.at(i).z;
        points3d_eig(3,i)=1;
    }

    ceres::Problem problem;

    for(int i = 0; i < tri.mworldMapPointsV.size(); i++)
    {
        ceres::CostFunction* cost_function=
            new ceres::AutoDiffCostFunction<mvo::SnavelyReprojectionError, 2, 3, 3>
            (new mvo::SnavelyReprojectionError(points2d_eig(0,i), points2d_eig(1,i), points3d_eig.col(i), focald, camX, camY));
        ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(cost_function, loss, rvec_eig.data(), tvec_eig.data());
    }

    ceres::Solver::Options options;
    
    options.linear_solver_type = ceres::DENSE_SCHUR;
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