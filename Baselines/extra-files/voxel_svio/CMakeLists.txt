cmake_minimum_required(VERSION 3.16)
project(vslamlab_voxel_svio LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
find_package(Ceres 2.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system date_time)
find_package(yaml-cpp REQUIRED)
file(GLOB estimator_sources CONFIGURE_DEPENDS "${VOXEL_SOURCE}/src/*.cpp")
add_library(voxel_svio_core STATIC ${estimator_sources})
target_include_directories(voxel_svio_core PUBLIC
    ${VOXEL_SOURCE}/include ${CMAKE_CURRENT_SOURCE_DIR}
    ${VOXEL_SOURCE}/thirdLibrary/tessil-src/include
    ${PCL_HEADERS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(voxel_svio_core PUBLIC
    Eigen3::Eigen Ceres::ceres yaml-cpp ${OpenCV_LIBRARIES}
    Boost::filesystem Boost::system Boost::date_time pthread)
add_executable(vslamlab_voxel_svio_stereo_vi runner.cpp)
target_link_libraries(vslamlab_voxel_svio_stereo_vi PRIVATE voxel_svio_core)
enable_testing()
add_executable(voxel_svio_contract_test contract_test.cpp)
target_link_libraries(voxel_svio_contract_test PRIVATE voxel_svio_core)
add_test(NAME voxel_svio_contract_test COMMAND voxel_svio_contract_test)
