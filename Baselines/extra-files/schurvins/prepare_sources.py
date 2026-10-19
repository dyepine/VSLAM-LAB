"""Replace ROS parameter access in generated factory copies; preserve core sources."""
from pathlib import Path
import sys

def write_changed(path,content):
    if not path.is_file() or path.read_text()!=content:path.write_text(content)

s=Path(sys.argv[1]);g=s/'build/generated'
(g/'include/svo_ros').mkdir(parents=True,exist_ok=True)
(g/'include/vikit').mkdir(parents=True,exist_ok=True)
write_changed(g/'include/vikit/params_helper.h','#include "file_parameters.h"\n')
header=(s/'svo_ros/include/svo_ros/svo_factory.h').read_text().replace('#include <ros/ros.h>','#include "file_parameters.h"')
write_changed(g/'include/svo_ros/svo_factory.h',header)
source=(s/'svo_ros/src/svo_factory.cpp').read_text().replace('#include <ros/package.h>','#include "file_parameters.h"')
write_changed(g/'svo_factory.cpp',source)
