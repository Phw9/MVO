# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/phw93/Dev/MVO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/phw93/Dev/MVO/build

# Include any dependencies generated for this target.
include CMakeFiles/mine.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mine.dir/flags.make

CMakeFiles/mine.dir/src/Feature.cpp.o: CMakeFiles/mine.dir/flags.make
CMakeFiles/mine.dir/src/Feature.cpp.o: ../src/Feature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mine.dir/src/Feature.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mine.dir/src/Feature.cpp.o -c /home/phw93/Dev/MVO/src/Feature.cpp

CMakeFiles/mine.dir/src/Feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mine.dir/src/Feature.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw93/Dev/MVO/src/Feature.cpp > CMakeFiles/mine.dir/src/Feature.cpp.i

CMakeFiles/mine.dir/src/Feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mine.dir/src/Feature.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw93/Dev/MVO/src/Feature.cpp -o CMakeFiles/mine.dir/src/Feature.cpp.s

CMakeFiles/mine.dir/src/PoseEstimation.cpp.o: CMakeFiles/mine.dir/flags.make
CMakeFiles/mine.dir/src/PoseEstimation.cpp.o: ../src/PoseEstimation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mine.dir/src/PoseEstimation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mine.dir/src/PoseEstimation.cpp.o -c /home/phw93/Dev/MVO/src/PoseEstimation.cpp

CMakeFiles/mine.dir/src/PoseEstimation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mine.dir/src/PoseEstimation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw93/Dev/MVO/src/PoseEstimation.cpp > CMakeFiles/mine.dir/src/PoseEstimation.cpp.i

CMakeFiles/mine.dir/src/PoseEstimation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mine.dir/src/PoseEstimation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw93/Dev/MVO/src/PoseEstimation.cpp -o CMakeFiles/mine.dir/src/PoseEstimation.cpp.s

CMakeFiles/mine.dir/src/Triangulate.cpp.o: CMakeFiles/mine.dir/flags.make
CMakeFiles/mine.dir/src/Triangulate.cpp.o: ../src/Triangulate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/mine.dir/src/Triangulate.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mine.dir/src/Triangulate.cpp.o -c /home/phw93/Dev/MVO/src/Triangulate.cpp

CMakeFiles/mine.dir/src/Triangulate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mine.dir/src/Triangulate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw93/Dev/MVO/src/Triangulate.cpp > CMakeFiles/mine.dir/src/Triangulate.cpp.i

CMakeFiles/mine.dir/src/Triangulate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mine.dir/src/Triangulate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw93/Dev/MVO/src/Triangulate.cpp -o CMakeFiles/mine.dir/src/Triangulate.cpp.s

CMakeFiles/mine.dir/src/Init.cpp.o: CMakeFiles/mine.dir/flags.make
CMakeFiles/mine.dir/src/Init.cpp.o: ../src/Init.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/mine.dir/src/Init.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mine.dir/src/Init.cpp.o -c /home/phw93/Dev/MVO/src/Init.cpp

CMakeFiles/mine.dir/src/Init.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mine.dir/src/Init.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw93/Dev/MVO/src/Init.cpp > CMakeFiles/mine.dir/src/Init.cpp.i

CMakeFiles/mine.dir/src/Init.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mine.dir/src/Init.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw93/Dev/MVO/src/Init.cpp -o CMakeFiles/mine.dir/src/Init.cpp.s

# Object files for target mine
mine_OBJECTS = \
"CMakeFiles/mine.dir/src/Feature.cpp.o" \
"CMakeFiles/mine.dir/src/PoseEstimation.cpp.o" \
"CMakeFiles/mine.dir/src/Triangulate.cpp.o" \
"CMakeFiles/mine.dir/src/Init.cpp.o"

# External object files for target mine
mine_EXTERNAL_OBJECTS =

libmine.so: CMakeFiles/mine.dir/src/Feature.cpp.o
libmine.so: CMakeFiles/mine.dir/src/PoseEstimation.cpp.o
libmine.so: CMakeFiles/mine.dir/src/Triangulate.cpp.o
libmine.so: CMakeFiles/mine.dir/src/Init.cpp.o
libmine.so: CMakeFiles/mine.dir/build.make
libmine.so: /usr/lib/x86_64-linux-gnu/libglut.so
libmine.so: /usr/lib/x86_64-linux-gnu/libXmu.so
libmine.so: /usr/lib/x86_64-linux-gnu/libXi.so
libmine.so: /usr/local/lib/libopencv_gapi.so.4.4.0
libmine.so: /usr/local/lib/libopencv_stitching.so.4.4.0
libmine.so: /usr/local/lib/libopencv_alphamat.so.4.4.0
libmine.so: /usr/local/lib/libopencv_aruco.so.4.4.0
libmine.so: /usr/local/lib/libopencv_bgsegm.so.4.4.0
libmine.so: /usr/local/lib/libopencv_bioinspired.so.4.4.0
libmine.so: /usr/local/lib/libopencv_ccalib.so.4.4.0
libmine.so: /usr/local/lib/libopencv_dnn_objdetect.so.4.4.0
libmine.so: /usr/local/lib/libopencv_dnn_superres.so.4.4.0
libmine.so: /usr/local/lib/libopencv_dpm.so.4.4.0
libmine.so: /usr/local/lib/libopencv_face.so.4.4.0
libmine.so: /usr/local/lib/libopencv_freetype.so.4.4.0
libmine.so: /usr/local/lib/libopencv_fuzzy.so.4.4.0
libmine.so: /usr/local/lib/libopencv_hfs.so.4.4.0
libmine.so: /usr/local/lib/libopencv_img_hash.so.4.4.0
libmine.so: /usr/local/lib/libopencv_intensity_transform.so.4.4.0
libmine.so: /usr/local/lib/libopencv_line_descriptor.so.4.4.0
libmine.so: /usr/local/lib/libopencv_quality.so.4.4.0
libmine.so: /usr/local/lib/libopencv_rapid.so.4.4.0
libmine.so: /usr/local/lib/libopencv_reg.so.4.4.0
libmine.so: /usr/local/lib/libopencv_rgbd.so.4.4.0
libmine.so: /usr/local/lib/libopencv_saliency.so.4.4.0
libmine.so: /usr/local/lib/libopencv_sfm.so.4.4.0
libmine.so: /usr/local/lib/libopencv_stereo.so.4.4.0
libmine.so: /usr/local/lib/libopencv_structured_light.so.4.4.0
libmine.so: /usr/local/lib/libopencv_superres.so.4.4.0
libmine.so: /usr/local/lib/libopencv_surface_matching.so.4.4.0
libmine.so: /usr/local/lib/libopencv_tracking.so.4.4.0
libmine.so: /usr/local/lib/libopencv_videostab.so.4.4.0
libmine.so: /usr/local/lib/libopencv_xfeatures2d.so.4.4.0
libmine.so: /usr/local/lib/libopencv_xobjdetect.so.4.4.0
libmine.so: /usr/local/lib/libopencv_xphoto.so.4.4.0
libmine.so: /usr/lib/x86_64-linux-gnu/libGL.so
libmine.so: /usr/lib/x86_64-linux-gnu/libGLU.so
libmine.so: /usr/local/lib/libpango_glgeometry.so
libmine.so: /usr/local/lib/libpango_plot.so
libmine.so: /usr/local/lib/libpango_python.so
libmine.so: /usr/local/lib/libpango_scene.so
libmine.so: /usr/local/lib/libpango_tools.so
libmine.so: /usr/local/lib/libpango_video.so
libmine.so: ../thirdparty/DBoW2/lib/libDBoW2.so
libmine.so: ../thirdparty/g2o/lib/libg2o.so
libmine.so: /usr/local/lib/libopencv_highgui.so.4.4.0
libmine.so: /usr/local/lib/libopencv_shape.so.4.4.0
libmine.so: /usr/local/lib/libopencv_datasets.so.4.4.0
libmine.so: /usr/local/lib/libopencv_plot.so.4.4.0
libmine.so: /usr/local/lib/libopencv_text.so.4.4.0
libmine.so: /usr/local/lib/libopencv_dnn.so.4.4.0
libmine.so: /usr/local/lib/libopencv_ml.so.4.4.0
libmine.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.4.0
libmine.so: /usr/local/lib/libopencv_optflow.so.4.4.0
libmine.so: /usr/local/lib/libopencv_ximgproc.so.4.4.0
libmine.so: /usr/local/lib/libopencv_video.so.4.4.0
libmine.so: /usr/local/lib/libopencv_videoio.so.4.4.0
libmine.so: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
libmine.so: /usr/local/lib/libopencv_objdetect.so.4.4.0
libmine.so: /usr/local/lib/libopencv_calib3d.so.4.4.0
libmine.so: /usr/local/lib/libopencv_features2d.so.4.4.0
libmine.so: /usr/local/lib/libopencv_flann.so.4.4.0
libmine.so: /usr/local/lib/libopencv_photo.so.4.4.0
libmine.so: /usr/local/lib/libopencv_imgproc.so.4.4.0
libmine.so: /usr/local/lib/libopencv_core.so.4.4.0
libmine.so: /usr/local/lib/libpango_geometry.so
libmine.so: /usr/local/lib/libtinyobj.so
libmine.so: /usr/local/lib/libpango_display.so
libmine.so: /usr/local/lib/libpango_vars.so
libmine.so: /usr/local/lib/libpango_windowing.so
libmine.so: /usr/local/lib/libpango_opengl.so
libmine.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
libmine.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
libmine.so: /usr/lib/x86_64-linux-gnu/libGLX.so
libmine.so: /usr/lib/x86_64-linux-gnu/libGLU.so
libmine.so: /usr/local/lib/libpango_image.so
libmine.so: /usr/local/lib/libpango_packetstream.so
libmine.so: /usr/local/lib/libpango_core.so
libmine.so: CMakeFiles/mine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libmine.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mine.dir/build: libmine.so

.PHONY : CMakeFiles/mine.dir/build

CMakeFiles/mine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mine.dir/clean

CMakeFiles/mine.dir/depend:
	cd /home/phw93/Dev/MVO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phw93/Dev/MVO /home/phw93/Dev/MVO /home/phw93/Dev/MVO/build /home/phw93/Dev/MVO/build /home/phw93/Dev/MVO/build/CMakeFiles/mine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mine.dir/depend

