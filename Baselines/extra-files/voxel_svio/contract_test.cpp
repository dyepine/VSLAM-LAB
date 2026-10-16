#include "mapManagement.h"
#include "quatOps.h"
#include <stdexcept>

int main() {
    voxelHashMap map;
    auto history = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    auto first = std::make_shared<mapPoint>();
    auto second = std::make_shared<mapPoint>();
    first->setPointXYZ(Eigen::Vector3d(0.1,0.1,0.1),false);
    second->setPointXYZ(Eigen::Vector3d(0.2,0.2,0.2),false);
    if (!mapManagement::addPointToVoxel(map,first,1.0,1,0.0,history))
        throw std::runtime_error("First point was not inserted");
    if (mapManagement::addPointToVoxel(map,second,1.0,1,0.0,history))
        throw std::runtime_error("Full voxel falsely accepted a second point");
    if (map.at(voxel(0,0,0)).NumPoints()!=1)
        throw std::runtime_error("Full voxel capacity changed");
    // The JPL state rotation is R_GtoI; Hamilton pose orientation is its transpose.
    Eigen::Matrix3d R_GI = Eigen::AngleAxisd(0.7,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    jplQuat state;
    state.setValue(quatType::rotToQuat(R_GI));
    Eigen::Quaterniond exported(state.getRot().transpose());
    if ((exported.toRotationMatrix()-R_GI.transpose()).norm()>1e-12)
        throw std::runtime_error("Incorrect JPL to Hamilton pose orientation");
    return 0;
}
