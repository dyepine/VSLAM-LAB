#pragma once
#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>
namespace ros {
class NodeHandle {
 public:
  explicit NodeHandle(const std::string& file): values(YAML::LoadFile(file)) {}
  YAML::Node values;
};
namespace package { inline std::string getPath(const std::string&) { return "."; } }
}
namespace vk {
template<class T> T param(const ros::NodeHandle& p, const std::string& key, const T& fallback, bool = false) {
 const YAML::Node n=p.values[key]; return n.IsDefined() ? n.as<T>() : fallback;
}
}
