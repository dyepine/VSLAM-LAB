// VSLAM-LAB I/O only. Estimation is performed by the original OKVIS 1 library.
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glog/logging.h>
#include <condition_variable>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

okvis::Time timeNs(int64_t ns) {
  return okvis::Time(ns / 1000000000LL, ns % 1000000000LL);
}

double sensorValue(const std::string& text) {
  // std::stod throws on representable subnormal BotanicGarden samples (ERANGE).
  // strtod preserves them; still reject malformed and non-finite measurements.
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str()) throw std::runtime_error("Invalid IMU number");
  while (std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end || !std::isfinite(value)) throw std::runtime_error("Invalid/non-finite IMU number");
  return value;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  if (argc != 5) {
    std::cerr << "Usage: runner settings.yaml images.tsv imu.csv trajectory.csv\n";
    return 2;
  }
  try {
    okvis::VioParametersReader reader(argv[1]);
    okvis::VioParameters params;
    reader.getParameters(params);
    params.visualization.displayImages = false;
    params.publishing.publishImuPropagatedState = false;
    if (params.nCameraSystem.numCameras() != 2)
      throw std::runtime_error("Exactly two cameras required");
    std::ifstream images(argv[2]), imu(argv[3]);
    std::ofstream output(argv[4]);
    if (!images || !imu || !output) throw std::runtime_error("Cannot open input/output");
    output << "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n" << std::setprecision(17);
    std::mutex mutex;
    std::condition_variable published;
    int64_t latest = -1, finalStamp = -1;
    size_t count = 0, frames = 0;
    {
      okvis::ThreadedKFVio estimator(params);
      estimator.setBlocking(true);
      estimator.setFullStateCallback([&](const okvis::Time& t,
          const okvis::kinematics::Transformation& pose,
          const Eigen::Matrix<double, 9, 1>&, const Eigen::Vector3d&) {
        std::lock_guard<std::mutex> lock(mutex);
        const int64_t ns = int64_t(t.sec) * 1000000000LL + t.nsec;
        // Library callbacks are T_WS; the catalog protocol expects body T_WB.
        const auto bodyPose = pose * params.imu.T_BS.inverse();
        const auto p = bodyPose.r();
        const auto q = bodyPose.q();
        output << ns << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
               << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
        latest = ns;
        ++count;
        published.notify_all();
      });
      std::string line;
      std::getline(imu, line); // header
      int64_t lastImu = -1;
      while (std::getline(images, line)) {
        std::stringstream row(line);
        std::string leftStamp, leftPath, rightStamp, rightPath;
        std::getline(row, leftStamp, '\t'); std::getline(row, leftPath, '\t');
        std::getline(row, rightStamp, '\t'); std::getline(row, rightPath);
        const int64_t tl = std::stoll(leftStamp), tr = std::stoll(rightStamp);
        const int64_t until = std::max(tl, tr) + 30000000LL;
        while (lastImu <= until) {
          if (!std::getline(imu, line)) throw std::runtime_error("IMU ends before camera + 30ms overlap");
          std::stringstream sample(line);
          std::string value;
          std::getline(sample, value, ',');
          lastImu = std::stoll(value);
          Eigen::Vector3d gyro, accel;
          for (int i = 0; i < 3; ++i) {
            if (!std::getline(sample, value, ',')) throw std::runtime_error("Incomplete IMU row");
            gyro[i] = sensorValue(value);
          }
          for (int i = 0; i < 3; ++i) {
            if (!std::getline(sample, value, ',')) throw std::runtime_error("Incomplete IMU row");
            accel[i] = sensorValue(value);
          }
          estimator.addImuMeasurement(timeNs(lastImu), accel, gyro);
        }
        auto left = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
        auto right = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) throw std::runtime_error("Cannot decode stereo images");
        estimator.addImage(timeNs(tl), 0, left);
        estimator.addImage(timeNs(tr), 1, right);
        // FrameSynchronizer averages the pair; its integer rounding can vary by 1ns.
        finalStamp = std::min(tl, tr);
        if (++frames % 50 == 0) std::cout << "Stereo frames: " << frames << std::endl;
      }
      std::unique_lock<std::mutex> lock(mutex);
      if (!published.wait_for(lock, std::chrono::seconds(60), [&] { return latest >= finalStamp; }))
        throw std::runtime_error("Timed out waiting for final optimized stereo pose");
    } // join publisher before closing the trajectory
    output.close();
    if (count < 2) throw std::runtime_error("Fewer than two optimized poses");
    std::cout << "Optimized poses: " << count << ", stereo frames: " << frames << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
