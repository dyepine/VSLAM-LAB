#pragma once
#include <yaml-cpp/yaml.h>
#include <sstream>
#include <string>

// File-backed equivalent of NodeHandle::param for the unchanged parameter mapping.
class FileParameters {
    YAML::Node root;
public:
    explicit FileParameters(const std::string& path) : root(YAML::LoadFile(path)) {}
    template<class T> void param(const std::string& key, T& destination, const T& fallback) const {
        YAML::Node node = root;
        std::stringstream path(key);
        std::string part;
        while (std::getline(path, part, '/')) {
            if (!node.IsMap() || !node[part].IsDefined()) { destination = fallback; return; }
            node.reset(node[part]);
        }
        destination = node.as<T>();
    }
};
