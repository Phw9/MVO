#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include "ceres/ceres.h"
#include "ceres/loss_function.h"


/*
    SnavelyReprojectionError :: CostFunctor
    bool operator() (camera, point, residuals)
        camera : 9 parameters
            0-2 parameters -> camera rotation vector (Rodriguez)
            3-5 parameters -> camera translation vector
            6   parameters -> focal length
            7-8 parameters -> radial distortion
        3d point : 3 parameters
        residuals : reprojection error
*/



