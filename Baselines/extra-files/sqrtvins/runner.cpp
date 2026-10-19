// File-input driver for the unmodified pinned rpng/sqrtVINS implementation.
#include "core/VioManager.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/sensor_data.h"
#include "utils/print.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: runner config.yaml images.tsv imu.csv trajectory.csv origin_ns\n";
        return 2;
    }
    try {
        auto parser = std::make_shared<ov_core::YamlParser>(argv[1]);
        ov_core::Printer::setPrintLevel("WARNING");
        ov_srvins::VioManagerOptions params;
        params.print_and_load(parser);
        params.num_opencv_threads = 1;
        params.use_multi_threading_subs = false;
        params.use_multi_threading_pubs = false;
        if (!parser->successful()) throw std::runtime_error("Invalid sqrtVINS configuration");
        ov_srvins::VioManager estimator(params);
        const int64_t origin = std::stoll(argv[5]);
        std::ifstream images(argv[2]), imu(argv[3]);
        std::ofstream trajectory(argv[4]);
        if (!images || !imu || !trajectory) throw std::runtime_error("Cannot open inputs/output");
        trajectory << "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n" << std::setprecision(17);
        std::string line; std::getline(imu, line);
        int64_t last_imu = -1, previous_stamp = -1;
        size_t frames = 0, poses = 0;
        while (std::getline(images, line)) {
            std::stringstream row(line);
            std::string stamp_text, left, right_stamp_text, right;
            std::getline(row, stamp_text, '\t'); std::getline(row, left, '\t');
            std::getline(row, right_stamp_text, '\t'); std::getline(row, right);
            const int64_t stamp = std::stoll(stamp_text), right_stamp = std::stoll(right_stamp_text);
            if (stamp <= previous_stamp) throw std::runtime_error("Non-monotonic camera input");
            previous_stamp = stamp;
            while (last_imu <= std::max(stamp, right_stamp)) {
                if (!std::getline(imu, line)) throw std::runtime_error("IMU ends before stereo frame");
                std::stringstream sample(line); std::string value;
                std::getline(sample, value, ',');
                const int64_t next = std::stoll(value);
                if (next <= last_imu) throw std::runtime_error("Non-monotonic IMU input");
                last_imu = next;
                ov_core::ImuData measurement;
                measurement.timestamp = (last_imu - origin) * 1e-9;
                for (int i = 0; i < 6; ++i) {
                    if (!std::getline(sample, value, ',')) throw std::runtime_error("Incomplete IMU sample");
                    double v = std::stod(value);
                    if (!std::isfinite(v)) throw std::runtime_error("Non-finite IMU sample");
                    if (i < 3) measurement.wm[i] = v; else measurement.am[i-3] = v;
                }
                estimator.feed_measurement_imu(measurement);
            }
            ov_core::CameraData camera;
            camera.timestamp = (stamp - origin) * 1e-9;
            camera.sensor_ids = {0, 1};
            camera.images = {cv::imread(left, cv::IMREAD_GRAYSCALE), cv::imread(right, cv::IMREAD_GRAYSCALE)};
            for (const auto& image : camera.images) {
                if (image.empty()) throw std::runtime_error("Cannot decode stereo image");
                camera.masks.push_back(cv::Mat::zeros(image.rows, image.cols, CV_8UC1));
            }
            estimator.feed_measurement_camera(camera);
            auto state = estimator.get_state();
            if (estimator.initialized() && std::abs(state->timestamp-camera.timestamp) < 1e-6) {
                // Native JPL R_GtoI -> Hamilton R_WI; output is the IMU pose.
                Eigen::Matrix3d rotation = state->imu->Rot().transpose().cast<double>();
                Eigen::Quaterniond q(rotation); const auto p = state->imu->pos();
                trajectory << stamp << ',' << p[0] << ',' << p[1] << ',' << p[2] << ','
                           << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
                ++poses;
            }
            if (++frames % 50 == 0) std::cout << "Stereo frames: " << frames << ", initialized poses: " << poses << std::endl;
        }
        trajectory.close();
        if (!trajectory || poses < 3) throw std::runtime_error("sqrtVINS did not export sufficient poses");
        std::cout << "Completed stereo pairs: " << frames << ", initialized poses: " << poses << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
