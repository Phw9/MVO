#pragma once
/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

// DBoW2 // defines OrbVocabulary and OrbDatabase
#include "MapData.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
// #include "g2o/solvers/csparse/linear_solver_csparse.h"
// #include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
// #include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/factory.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
// #include "g2o/types/sba/types_six_dof_expmap.h"


namespace mvo
{
    namespace LoopClosure
    {
        void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);
        void wait();
    // ----------------------------------------------------------------------------
        void LoopDetectCompute(const cv::Mat& img, std::vector<std::vector<cv::Mat>>& globaldesc, OrbDatabase& db);

    // ----------------------------------------------------------------------------
        void VocCreation(const std::vector<std::vector<cv::Mat>>& features);

    // ----------------------------------------------------------------------------
        void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);

    } // LoopClosure

    namespace PGO
    {

    } // PGO
} // mvo

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);

