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
CMAKE_SOURCE_DIR = /home/phw9/Dev/MVO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/phw9/Dev/MVO/build

# Include any dependencies generated for this target.
include CMakeFiles/mvo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mvo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mvo.dir/flags.make

CMakeFiles/mvo.dir/main/main.cpp.o: CMakeFiles/mvo.dir/flags.make
CMakeFiles/mvo.dir/main/main.cpp.o: ../main/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw9/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mvo.dir/main/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mvo.dir/main/main.cpp.o -c /home/phw9/Dev/MVO/main/main.cpp

CMakeFiles/mvo.dir/main/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mvo.dir/main/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw9/Dev/MVO/main/main.cpp > CMakeFiles/mvo.dir/main/main.cpp.i

CMakeFiles/mvo.dir/main/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mvo.dir/main/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw9/Dev/MVO/main/main.cpp -o CMakeFiles/mvo.dir/main/main.cpp.s

CMakeFiles/mvo.dir/main/Init.cpp.o: CMakeFiles/mvo.dir/flags.make
CMakeFiles/mvo.dir/main/Init.cpp.o: ../main/Init.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw9/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mvo.dir/main/Init.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mvo.dir/main/Init.cpp.o -c /home/phw9/Dev/MVO/main/Init.cpp

CMakeFiles/mvo.dir/main/Init.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mvo.dir/main/Init.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw9/Dev/MVO/main/Init.cpp > CMakeFiles/mvo.dir/main/Init.cpp.i

CMakeFiles/mvo.dir/main/Init.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mvo.dir/main/Init.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw9/Dev/MVO/main/Init.cpp -o CMakeFiles/mvo.dir/main/Init.cpp.s

# Object files for target mvo
mvo_OBJECTS = \
"CMakeFiles/mvo.dir/main/main.cpp.o" \
"CMakeFiles/mvo.dir/main/Init.cpp.o"

# External object files for target mvo
mvo_EXTERNAL_OBJECTS =

mvo: CMakeFiles/mvo.dir/main/main.cpp.o
mvo: CMakeFiles/mvo.dir/main/Init.cpp.o
mvo: CMakeFiles/mvo.dir/build.make
mvo: ../src/Feature.cpp
mvo: ../src/KeyFrame.cpp
mvo: ../src/PoseEstimation.cpp
mvo: ../src/Triangulate.cpp
mvo: /usr/lib/x86_64-linux-gnu/libglut.so
mvo: /usr/lib/x86_64-linux-gnu/libXi.so
mvo: /usr/local/lib/libopencv_shape.so.3.2.0
mvo: /usr/local/lib/libopencv_stitching.so.3.2.0
mvo: /usr/local/lib/libopencv_superres.so.3.2.0
mvo: /usr/local/lib/libopencv_videostab.so.3.2.0
mvo: /usr/lib/x86_64-linux-gnu/libGL.so
mvo: /usr/lib/x86_64-linux-gnu/libGLU.so
mvo: /usr/local/lib/libpango_glgeometry.so
mvo: /usr/local/lib/libpango_plot.so
mvo: /usr/local/lib/libpango_scene.so
mvo: /usr/local/lib/libpango_tools.so
mvo: /usr/local/lib/libpango_video.so
mvo: /usr/local/lib/libpango_display.so
mvo: /usr/local/lib/libpango_python.so
mvo: /usr/local/lib/libopencv_objdetect.so.3.2.0
mvo: /usr/local/lib/libopencv_calib3d.so.3.2.0
mvo: /usr/local/lib/libopencv_features2d.so.3.2.0
mvo: /usr/local/lib/libopencv_flann.so.3.2.0
mvo: /usr/local/lib/libopencv_highgui.so.3.2.0
mvo: /usr/local/lib/libopencv_ml.so.3.2.0
mvo: /usr/local/lib/libopencv_photo.so.3.2.0
mvo: /usr/local/lib/libopencv_video.so.3.2.0
mvo: /usr/local/lib/libopencv_videoio.so.3.2.0
mvo: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
mvo: /usr/local/lib/libopencv_imgproc.so.3.2.0
mvo: /usr/local/lib/libopencv_core.so.3.2.0
mvo: /usr/local/lib/libpango_geometry.so
mvo: /usr/local/lib/libtinyobj.so
mvo: /usr/local/lib/libpango_vars.so
mvo: /usr/local/lib/libpango_windowing.so
mvo: /usr/local/lib/libpango_opengl.so
mvo: /usr/lib/x86_64-linux-gnu/libGLEW.so
mvo: /usr/lib/x86_64-linux-gnu/libOpenGL.so
mvo: /usr/lib/x86_64-linux-gnu/libGLX.so
mvo: /usr/lib/x86_64-linux-gnu/libGLU.so
mvo: /usr/local/lib/libpango_image.so
mvo: /usr/local/lib/libpango_packetstream.so
mvo: /usr/local/lib/libpango_core.so
mvo: CMakeFiles/mvo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/phw9/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable mvo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mvo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mvo.dir/build: mvo

.PHONY : CMakeFiles/mvo.dir/build

CMakeFiles/mvo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mvo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mvo.dir/clean

CMakeFiles/mvo.dir/depend:
	cd /home/phw9/Dev/MVO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phw9/Dev/MVO /home/phw9/Dev/MVO /home/phw9/Dev/MVO/build /home/phw9/Dev/MVO/build /home/phw9/Dev/MVO/build/CMakeFiles/mvo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mvo.dir/depend

