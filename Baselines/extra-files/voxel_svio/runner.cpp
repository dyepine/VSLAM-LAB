#include "stereoVio.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>

double number(const std::string& text) {
    char* end = nullptr;
    double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(value)) throw std::runtime_error("Invalid IMU sample");
    while (std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end) throw std::runtime_error("Trailing data in IMU sample");
    return value;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: runner native.yaml images.tsv imu.csv trajectory.csv origin_ns\n";
        return 2;
    }
    try {
        const int64_t origin = std::stoll(argv[5]);
        std::ifstream images(argv[2]), imu(argv[3]);
        std::ofstream trajectory(argv[4]);
        if (!images || !imu || !trajectory) throw std::runtime_error("Cannot open inputs/output");
        trajectory << "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n" << std::setprecision(17);
        voxelStereoVio estimator(argv[1]);
        std::string line;
        std::getline(imu, line);
        int64_t lastImu = -1;
        size_t frames = 0, poses = 0;
        while (std::getline(images, line)) {
            std::stringstream row(line);
            std::string stampText, leftPath, rightStampText, rightPath;
            std::getline(row, stampText, '\t'); std::getline(row, leftPath, '\t');
            std::getline(row, rightStampText, '\t'); std::getline(row, rightPath);
            const int64_t stamp = std::stoll(stampText);
            const int64_t rightStamp = std::stoll(rightStampText);
            // Preserve the native run() contract: IMU must bracket the next camera stamp.
            while (lastImu <= std::max(stamp, rightStamp)) {
                if (!std::getline(imu, line)) throw std::runtime_error("IMU ends before stereo frame");
                std::stringstream sample(line);
                std::string value;
                std::getline(sample, value, ',');
                lastImu = std::stoll(value);
                imuData measurement;
                measurement.timestamp = (lastImu - origin) * 1e-9;
                for (int i = 0; i < 3; ++i) {
                    if (!std::getline(sample, value, ',')) throw std::runtime_error("Incomplete IMU row");
                    measurement.gyr[i] = number(value);
                }
                for (int i = 0; i < 3; ++i) {
                    if (!std::getline(sample, value, ',')) throw std::runtime_error("Incomplete IMU row");
                    measurement.acc[i] = number(value);
                }
                estimator.feedImu(measurement);
            }
            cameraData camera;
            camera.timestamp = (stamp - origin) * 1e-9;
            camera.camera_ids = {0, 1};
            camera.images = {cv::imread(leftPath, cv::IMREAD_GRAYSCALE), cv::imread(rightPath, cv::IMREAD_GRAYSCALE)};
            for (const auto& image : camera.images) {
                if (image.empty()) throw std::runtime_error("Cannot decode stereo image");
                camera.masks.push_back(cv::Mat::zeros(image.rows, image.cols, CV_8UC1));
            }
            estimator.feedStereo(camera);
            Eigen::Matrix4d T_WI;
            if (estimator.poseAt(camera.timestamp, T_WI)) {
                const Eigen::Vector3d p = T_WI.block<3,1>(0,3);
                const Eigen::Quaterniond q(T_WI.block<3,3>(0,0));
                trajectory << stamp << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
                           << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
                ++poses;
            }
            if (++frames % 50 == 0) std::cout << "Stereo frames: " << frames << ", initialized poses: " << poses << std::endl;
        }
        trajectory.close();
        if (!trajectory || !initial_flag || poses < 3) throw std::runtime_error("Voxel-SVIO did not initialize or export sufficient poses");
        std::cout << "Completed stereo pairs: " << frames << ", initialized poses: " << poses << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
