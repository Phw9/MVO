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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);
void wait();
// ----------------------------------------------------------------------------

void LoopDetectCompute(std::vector<std::vector<cv::Mat>>& globaldesc, std::vector<mvo::MapData>& map, OrbDatabase& db);

// ----------------------------------------------------------------------------

void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db);

// ----------------------------------------------------------------------------

