# Included at the end of the pinned upstream CMakeLists.txt by build.sh.
add_executable(vslamlab_okvis1_stereo_vi ${CMAKE_CURRENT_LIST_DIR}/runner.cpp)
target_compile_features(vslamlab_okvis1_stereo_vi PRIVATE cxx_std_14)
target_include_directories(vslamlab_okvis1_stereo_vi PRIVATE ${CMAKE_BINARY_DIR}/okvis_multisensor_processing)
target_link_libraries(vslamlab_okvis1_stereo_vi PRIVATE okvis_multisensor_processing ${OpenCV_LIBRARIES} pthread)
