cmake_minimum_required(VERSION 2.8)
project(mvo)

message(STATUS "${PROJECT_NAME}")
message(STATUS "${PROJECT_SOURCE_DIR}")
message(STATUS "${PROJECT_BINARY_DIR}")


find_package(OpenCV REQUIRED) 
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()

find_package(OpenGL REQUIRED)
   if(NOT OpenGL_FOUND)
      message(FATAL_ERROR "OpenGL not found.")
   endif()

find_package(g2o REQUIRED)
   if(NOT g2o_FOUND)
      message(FATAL_ERROR "g2o not found.")
   endif()

find_package(DBoW2 REQUIRED)
   if(NOT DBoW2_FOUND)
      message(FATAL_ERROR "DBoW2 not found.")
   endif()

find_package(GLUT REQUIRED)
   if(NOT GLUT_FOUND)
      message(FATAL_ERROR "GLUT not found.")
   endif()

find_package(Eigen3 REQUIRED)
   if(NOT Eigen3_FOUND)
      message(FATAL_ERROR "Eigen3 not found.")
   endif()

find_package(Pangolin REQUIRED)
   if(NOT Pangolin_FOUND)
      message(FATAL_ERROR "Pangolin not found.")
   endif()

find_package(Ceres REQUIRED)
   if(NOT Ceres_FOUND)
      message(FATAL_ERROR "Ceres-solver not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

# set(LIBRARY_OUTPUT_DIRECTORY "./lib")

include_directories(
   ${EIGEN3_INCLUDE_DIRS}
   ${GLUT_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS}
   ${OPENGL_INCLUDE_DIRS}
   ${Pangolin_INCLUDE_DIRS}
   ${DBoW2_INCLUDE_DIRS}
   ${g2o_INCLUDE_DIRS}
   ${CERES_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include
   )

add_library(mine SHARED
   ${PROJECT_SOURCE_DIR}/src/BundleAdjustment.cpp
   ${PROJECT_SOURCE_DIR}/src/Converter.cpp
   ${PROJECT_SOURCE_DIR}/src/Feature.cpp
   ${PROJECT_SOURCE_DIR}/src/Init.cpp
   ${PROJECT_SOURCE_DIR}/src/PoseEstimation.cpp
   ${PROJECT_SOURCE_DIR}/src/LoopClosure.cpp
   ${PROJECT_SOURCE_DIR}/src/Triangulate.cpp
   ${PROJECT_SOURCE_DIR}/src/MapData.cpp
)
target_compile_options(mine PUBLIC -g -Wall -std=c++17)

target_link_libraries(mine
   ${EIGEN3_LIBRARIES}
   ${GLUT_LIBRARY}
   ${OpenCV_LIBS}
   ${OPENGL_LIBRARIES}
   ${DBoW2_LIBRARIES}
   ${Pangolin_LIBRARIES}
   ${CERES_LIBRARIES}
   ${g2o_LIBRARIES}
   # ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib/libDBoW2.so
   # ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so
   )
   
add_executable(mvo
   ${PROJECT_SOURCE_DIR}/main.cpp
)

target_link_libraries(mvo
   mine
)

target_COMPILE_OPTIONS(mvo PUBLIC -g -Wall -std=c++17)


