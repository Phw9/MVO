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


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  std::cout << std::endl << "Press enter to continue" << std::endl;
  getchar();
}

// ----------------------------------------------------------------------------

void LoopDetectCompute(std::vector<std::vector<cv::Mat>>& globaldesc, std::vector<mvo::MapData>& map, OrbDatabase& db)
{
  wait();
  testDatabase(globaldesc, map, db);
}

// ----------------------------------------------------------------------------

void testDatabase(const std::vector<std::vector<cv::Mat>> &globaldesc, std::vector<mvo::MapData> map, OrbDatabase& db)
{
  std::cout << "Database information: " << std::endl << db << std::endl;

  // and query the database
  std::cout << "Querying the database: " << std::endl;

  int a = map.size();
  DBoW2::QueryResults ret;
  for(int i = 0; i < a; i++)
  {
    db.query(globaldesc[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    // std::cout << "Searching for Image " << i << ". " << ret << std::endl;
  }

  std::cout << std::endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  std::cout << "Saving database..." << std::endl;
  db.save("small_db.yml.gz");
  std::cout << "... done!" << std::endl;

  // once saved, we can load it again  
  std::cout << "Retrieving database once again..." << std::endl;
  OrbDatabase db2("small_db.yml.gz");
  std::cout << "... done! This is: " << std::endl << db2 << std::endl;
}

// ----------------------------------------------------------------------------

