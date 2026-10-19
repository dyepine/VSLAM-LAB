#include <svo_ros/svo_factory.h>
#include <svo/frame_handler_stereo.h>
#include <svo/imu_handler.h>
#include <svo/common/frame.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>
int main(int argc,char**argv) {
 if(argc!=6) {std::cerr<<"runner settings images.tsv imu.csv poses.csv origin_ns\n";return 2;}
 try {
  google::InitGoogleLogging(argv[0]); FLAGS_logtostderr=true;
  cv::setNumThreads(1);
  ros::NodeHandle p(argv[1]);
  auto vo=svo::factory::makeStereo(p); auto imu=svo::factory::getImuHandler(p);
  vo->setImuHandler(imu);vo->start();
  const int64_t origin=std::stoll(argv[5]);
  std::ifstream images(argv[2]),in(argv[3]);std::ofstream out(argv[4]);
  std::ofstream states(std::string(argv[4])+".tracking.csv");
  if(!images||!in||!out||!states)throw std::runtime_error("Cannot open input/output");
  out<<"ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n"<<std::setprecision(17);
  states<<"timestamp_ns,stage,exported\n";
  std::string line;std::getline(in,line);int64_t last=-1,previous=-1;size_t frames=0,poses=0;
  while(std::getline(images,line)) {
   std::stringstream row(line);std::string ts,left,rt,right;
   std::getline(row,ts,'\t');std::getline(row,left,'\t');std::getline(row,rt,'\t');std::getline(row,right);
   const int64_t stamp=std::stoll(ts),right_stamp=std::stoll(rt);
   if(stamp<=previous)throw std::runtime_error("Non-monotonic camera input");previous=stamp;
   while(last<=std::max(stamp,right_stamp)) {
    if(!std::getline(in,line))throw std::runtime_error("IMU ends before camera");
    std::stringstream sample(line);std::string v;std::getline(sample,v,',');int64_t t=std::stoll(v);
    if(t<=last)throw std::runtime_error("Non-monotonic IMU");last=t;
    Eigen::Vector3d w,a;
    for(int k=0;k<6;++k){if(!std::getline(sample,v,','))throw std::runtime_error("Incomplete IMU");double n=std::stod(v);if(!std::isfinite(n))throw std::runtime_error("Non-finite IMU");if(k<3)w[k]=n;else a[k-3]=n;}
    imu->addImuMeasurement(svo::ImuMeasurement((t-origin)*1e-9,w,a));
   }
   bool ready=true;svo::Quaternion rotation;
   const double t=(stamp-origin)*1e-9;
   if(!vo->hasStarted()) {ready=imu->getInitialAttitude(t,rotation);if(ready)vo->setRotationPrior(rotation);}
   else if(vo->getLastFrames()) {
    if(imu->getRelativeRotationPrior(vo->getLastFrames()->getMinTimestampSeconds(),t,false,rotation))vo->setRotationIncrementPrior(rotation);
   }
   bool exported=false;
   if(ready) {
    std::vector<cv::Mat> pictures={cv::imread(left,cv::IMREAD_GRAYSCALE),cv::imread(right,cv::IMREAD_GRAYSCALE)};
    for(const auto&im:pictures)if(im.empty())throw std::runtime_error("Cannot decode camera image");
    vo->addImageBundle(pictures,stamp-origin);
    if(vo->stage()==svo::Stage::kTracking&&vo->getLastFrames()&&vo->getLastFrames()->getMinTimestampNanoseconds()==stamp-origin) {
     auto T=vo->getLastFrames()->get_T_W_B();auto v=T.getPosition();auto q=T.getRotation().toImplementation();
     out<<stamp<<','<<v.x()<<','<<v.y()<<','<<v.z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<'\n';++poses;exported=true;
    }
    if(vo->stage()==svo::Stage::kPaused&&vk::param<bool>(p,"automatic_reinitialization",true))vo->start();
   }
   states<<stamp<<','<<vo->stageStr()<<','<<exported<<'\n';
   if(++frames%50==0){out.flush();states.flush();std::cout<<"Stereo frames: "<<frames<<", exported poses: "<<poses<<std::endl;}
  }
  std::cout<<"Completed stereo pairs: "<<frames<<", exported poses: "<<poses<<std::endl;
  if(poses<3)throw std::runtime_error("Insufficient initialized SchurVINS poses");
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
