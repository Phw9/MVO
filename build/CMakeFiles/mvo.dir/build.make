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
include CMakeFiles/mvo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mvo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mvo.dir/flags.make

CMakeFiles/mvo.dir/main.cpp.o: CMakeFiles/mvo.dir/flags.make
CMakeFiles/mvo.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mvo.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mvo.dir/main.cpp.o -c /home/phw93/Dev/MVO/main.cpp

CMakeFiles/mvo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mvo.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/phw93/Dev/MVO/main.cpp > CMakeFiles/mvo.dir/main.cpp.i

CMakeFiles/mvo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mvo.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/phw93/Dev/MVO/main.cpp -o CMakeFiles/mvo.dir/main.cpp.s

# Object files for target mvo
mvo_OBJECTS = \
"CMakeFiles/mvo.dir/main.cpp.o"

# External object files for target mvo
mvo_EXTERNAL_OBJECTS =

mvo: CMakeFiles/mvo.dir/main.cpp.o
mvo: CMakeFiles/mvo.dir/build.make
mvo: libmine.so
mvo: /usr/lib/x86_64-linux-gnu/libglut.so
mvo: /usr/lib/x86_64-linux-gnu/libXmu.so
mvo: /usr/lib/x86_64-linux-gnu/libXi.so
mvo: /usr/local/lib/libopencv_gapi.so.4.4.0
mvo: /usr/local/lib/libopencv_stitching.so.4.4.0
mvo: /usr/local/lib/libopencv_alphamat.so.4.4.0
mvo: /usr/local/lib/libopencv_aruco.so.4.4.0
mvo: /usr/local/lib/libopencv_bgsegm.so.4.4.0
mvo: /usr/local/lib/libopencv_bioinspired.so.4.4.0
mvo: /usr/local/lib/libopencv_ccalib.so.4.4.0
mvo: /usr/local/lib/libopencv_dnn_objdetect.so.4.4.0
mvo: /usr/local/lib/libopencv_dnn_superres.so.4.4.0
mvo: /usr/local/lib/libopencv_dpm.so.4.4.0
mvo: /usr/local/lib/libopencv_highgui.so.4.4.0
mvo: /usr/local/lib/libopencv_face.so.4.4.0
mvo: /usr/local/lib/libopencv_freetype.so.4.4.0
mvo: /usr/local/lib/libopencv_fuzzy.so.4.4.0
mvo: /usr/local/lib/libopencv_hfs.so.4.4.0
mvo: /usr/local/lib/libopencv_img_hash.so.4.4.0
mvo: /usr/local/lib/libopencv_intensity_transform.so.4.4.0
mvo: /usr/local/lib/libopencv_line_descriptor.so.4.4.0
mvo: /usr/local/lib/libopencv_quality.so.4.4.0
mvo: /usr/local/lib/libopencv_rapid.so.4.4.0
mvo: /usr/local/lib/libopencv_reg.so.4.4.0
mvo: /usr/local/lib/libopencv_rgbd.so.4.4.0
mvo: /usr/local/lib/libopencv_saliency.so.4.4.0
mvo: /usr/local/lib/libopencv_sfm.so.4.4.0
mvo: /usr/local/lib/libopencv_stereo.so.4.4.0
mvo: /usr/local/lib/libopencv_structured_light.so.4.4.0
mvo: /usr/local/lib/libopencv_phase_unwrapping.so.4.4.0
mvo: /usr/local/lib/libopencv_superres.so.4.4.0
mvo: /usr/local/lib/libopencv_optflow.so.4.4.0
mvo: /usr/local/lib/libopencv_surface_matching.so.4.4.0
mvo: /usr/local/lib/libopencv_tracking.so.4.4.0
mvo: /usr/local/lib/libopencv_datasets.so.4.4.0
mvo: /usr/local/lib/libopencv_plot.so.4.4.0
mvo: /usr/local/lib/libopencv_text.so.4.4.0
mvo: /usr/local/lib/libopencv_dnn.so.4.4.0
mvo: /usr/local/lib/libopencv_videostab.so.4.4.0
mvo: /usr/local/lib/libopencv_videoio.so.4.4.0
mvo: /usr/local/lib/libopencv_xfeatures2d.so.4.4.0
mvo: /usr/local/lib/libopencv_ml.so.4.4.0
mvo: /usr/local/lib/libopencv_shape.so.4.4.0
mvo: /usr/local/lib/libopencv_ximgproc.so.4.4.0
mvo: /usr/local/lib/libopencv_video.so.4.4.0
mvo: /usr/local/lib/libopencv_xobjdetect.so.4.4.0
mvo: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
mvo: /usr/local/lib/libopencv_objdetect.so.4.4.0
mvo: /usr/local/lib/libopencv_calib3d.so.4.4.0
mvo: /usr/local/lib/libopencv_features2d.so.4.4.0
mvo: /usr/local/lib/libopencv_flann.so.4.4.0
mvo: /usr/local/lib/libopencv_xphoto.so.4.4.0
mvo: /usr/local/lib/libopencv_photo.so.4.4.0
mvo: /usr/local/lib/libopencv_imgproc.so.4.4.0
mvo: /usr/local/lib/libopencv_core.so.4.4.0
mvo: /usr/lib/x86_64-linux-gnu/libGL.so
mvo: /usr/local/lib/libpango_glgeometry.so
mvo: /usr/local/lib/libpango_geometry.so
mvo: /usr/local/lib/libpango_plot.so
mvo: /usr/local/lib/libpango_python.so
mvo: /usr/local/lib/libpango_scene.so
mvo: /usr/local/lib/libpango_tools.so
mvo: /usr/local/lib/libpango_display.so
mvo: /usr/local/lib/libpango_vars.so
mvo: /usr/local/lib/libpango_video.so
mvo: /usr/local/lib/libpango_packetstream.so
mvo: /usr/local/lib/libpango_windowing.so
mvo: /usr/local/lib/libpango_opengl.so
mvo: /usr/local/lib/libpango_image.so
mvo: /usr/local/lib/libpango_core.so
mvo: /usr/lib/x86_64-linux-gnu/libGLEW.so
mvo: /usr/lib/x86_64-linux-gnu/libOpenGL.so
mvo: /usr/lib/x86_64-linux-gnu/libGLX.so
mvo: /usr/lib/x86_64-linux-gnu/libGLU.so
mvo: /usr/local/lib/libtinyobj.so
mvo: /usr/local/lib/libceres.a
mvo: /usr/lib/x86_64-linux-gnu/libglog.so
mvo: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
mvo: /usr/lib/x86_64-linux-gnu/libspqr.so
mvo: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
mvo: /usr/lib/x86_64-linux-gnu/libtbb.so
mvo: /usr/lib/x86_64-linux-gnu/libcholmod.so
mvo: /usr/lib/x86_64-linux-gnu/libccolamd.so
mvo: /usr/lib/x86_64-linux-gnu/libcamd.so
mvo: /usr/lib/x86_64-linux-gnu/libcolamd.so
mvo: /usr/lib/x86_64-linux-gnu/libamd.so
mvo: /usr/lib/x86_64-linux-gnu/liblapack.so
mvo: /usr/lib/x86_64-linux-gnu/libf77blas.so
mvo: /usr/lib/x86_64-linux-gnu/libatlas.so
mvo: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
mvo: /usr/lib/x86_64-linux-gnu/librt.so
mvo: /usr/lib/x86_64-linux-gnu/libcxsparse.so
mvo: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
mvo: /usr/lib/x86_64-linux-gnu/libtbb.so
mvo: /usr/lib/x86_64-linux-gnu/libcholmod.so
mvo: /usr/lib/x86_64-linux-gnu/libccolamd.so
mvo: /usr/lib/x86_64-linux-gnu/libcamd.so
mvo: /usr/lib/x86_64-linux-gnu/libcolamd.so
mvo: /usr/lib/x86_64-linux-gnu/libamd.so
mvo: /usr/lib/x86_64-linux-gnu/liblapack.so
mvo: /usr/lib/x86_64-linux-gnu/libf77blas.so
mvo: /usr/lib/x86_64-linux-gnu/libatlas.so
mvo: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
mvo: /usr/lib/x86_64-linux-gnu/librt.so
mvo: /usr/lib/x86_64-linux-gnu/libcxsparse.so
mvo: ../thirdparty/DBoW2/lib/libDBoW2.so
mvo: ../thirdparty/g2o/lib/libg2o.so
mvo: CMakeFiles/mvo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/phw93/Dev/MVO/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mvo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mvo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mvo.dir/build: mvo

.PHONY : CMakeFiles/mvo.dir/build

CMakeFiles/mvo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mvo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mvo.dir/clean

CMakeFiles/mvo.dir/depend:
	cd /home/phw93/Dev/MVO/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/phw93/Dev/MVO /home/phw93/Dev/MVO /home/phw93/Dev/MVO/build /home/phw93/Dev/MVO/build /home/phw93/Dev/MVO/build/CMakeFiles/mvo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mvo.dir/depend

