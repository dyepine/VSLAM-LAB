import os, cv2
from pathlib import Path
from typing import List

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

def _get_rgb_yaml_section(camera_params, sequence_name: str, dataset_path: Path) -> List[str]:
    """Generate YAML lines for rgb parameters."""
    # Get image dimensions
    sequence_path = os.path.join(dataset_path, sequence_name)
    rgb_path = os.path.join(sequence_path, camera_params['cam_name'])
    
    # Ensure rgb_path exists and has images before trying to read
    if not os.path.exists(rgb_path) or not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(rgb_path)):
        print(f"{SCRIPT_LABEL}Warning: RGB path {rgb_path} not found or no images present. Cannot determine image dimensions.")
        h, w = 0, 0 # Default or raise error
    else:
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not rgb_files:
            print(f"{SCRIPT_LABEL}Warning: No image files found in {rgb_path}. Cannot determine image dimensions.")
            h, w = 0,0 # Default or raise error
        else:
            image_0_path = os.path.join(rgb_path, rgb_files[0])
            image_0 = cv2.imread(image_0_path)
            if image_0 is None:
                print(f"{SCRIPT_LABEL}Warning: Could not read image {image_0_path}. Cannot determine image dimensions.")
                h,w = 0,0 # Default or raise error
            else:
                h, w, channels = image_0.shape

    lines = []
    lines.append(f"  - {{cam_name: {camera_params['cam_name']},")
    lines.append(f"     cam_type: {camera_params['cam_type']},")
    lines.append(f"     cam_model: {camera_params['cam_model']},")
    if 'distortion_type' in camera_params:
        lines.append(f"     distortion_type: {camera_params['distortion_type']},")
    lines.append(f"     focal_length: {camera_params['focal_length']},")
    lines.append(f"     principal_point: {camera_params['principal_point']},")
    if 'distortion_coefficients' in camera_params:
        lines.append(f"     distortion_coefficients: {camera_params['distortion_coefficients']}, ")

    lines.append(f"     image_dimension: [{w}, {h}],")
    lines.append(f"     fps: {camera_params['fps']},")

    T_BS_data = camera_params['T_BS']
    flat_list_str = [f"{x:.13f}" for x in T_BS_data.flatten()]
    lines.append("     T_BS: [" + ', '.join(flat_list_str) + "] # Sensor extrinsics wrt. the body-frame.")

    lines.append(f"    }}\n")
    return lines

def _get_rgbd_yaml_section(camera_params, sequence_name: str, dataset_path: Path) -> List[str]:
    """Generate YAML lines for rgb parameters."""
    # """Generate YAML lines for RGBD parameters."""
    lines = _get_rgb_yaml_section(camera_params, sequence_name, dataset_path);
    lines.insert(2, f"     depth_name: {camera_params['depth_name']},")
    lines.insert(8, f"     depth_factor: {camera_params['depth_factor']},")
    return lines

def _get_imu_yaml_section(imu_params) -> List[str]:
    # Optional source fields stay absent instead of inventing saturation or bias priors.
    fields = ["a_max", "g_max", "sigma_g_c", "sigma_gw_c", "sigma_a_c",
              "sigma_aw_c", "sigma_bg", "sigma_ba", "a0", "g0", "g", "s_a", "fps"]
    lines = [f"  - {{imu_name: {imu_params['imu_name']},"]
    for key in fields:
        if key in imu_params:
            value = imu_params[key]
            if key.startswith('sigma_'):
                value = f"{value:.6e}"
            lines.append(f"     {key}: {value},")
    values = ', '.join(f"{x:.13f}" for x in imu_params['T_BS'].flatten())
    lines.append(f"     T_BS: [{values}] # Sensor extrinsics wrt. the body-frame.")
    lines.append("    }\n")
    return lines
