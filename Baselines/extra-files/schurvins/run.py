"""File input and frame/calibration conversion for pinned author SchurVINS."""
import argparse,copy,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ADAPTER=Path(__file__).resolve().parent
SOURCE=ADAPTER.parent.parent/'SchurVINS'
sys.path.insert(0,str(ADAPTER.parent))
from stereo_vi_inputs import read_yaml,transform,write_manifest

def native_settings(calibration,settings,folder):
    imu=next(i for i in calibration['imus'] if i['imu_name']=='imu_0');T_BI=transform(imu['T_BS'])
    # Retain author saturation/initialization priors; replace actual sensor calibration.
    native=read_yaml(ADAPTER/'vslamlab_schurvins_imu_priors.yaml')
    native['label']='VSLAM-LAB';native['cameras']=[]
    for index in [0,1]:
        cam=next(c for c in calibration['cameras'] if c['cam_name']==f'rgb_{index}')
        d=cam['distortion_type']
        if cam['cam_model']!='pinhole' or d not in ['pinhole','radtan4','equid4']:
            raise ValueError(f'SchurVINS file adapter requires pinhole/radtan4/equid4, got {d}')
        coeff=[0.]*4 if d=='pinhole' else cam['distortion_coefficients']
        if len(coeff)!=4 or not np.isfinite(coeff).all():raise ValueError('Expected four finite distortion coefficients')
        w,h=cam['image_dimension'];K=cam['focal_length']+cam['principal_point']
        native['cameras'].append(dict(camera=dict(label=f'cam{index}',type='pinhole',image_width=w,image_height=h,
            **{'line-delay-nanoseconds':0},intrinsics=dict(rows=4,cols=1,data=K),
            distortion=dict(type='equidistant' if d=='equid4' else 'radial-tangential',parameters=dict(rows=4,cols=1,data=coeff))),
            T_B_C=dict(rows=4,cols=4,data=(np.linalg.inv(T_BI)@transform(cam['T_BS'])).ravel().tolist())))
    for src,dst in [('sigma_g_c','sigma_omega_c'),('sigma_a_c','sigma_acc_c'),('sigma_gw_c','sigma_omega_bias_c'),('sigma_aw_c','sigma_acc_bias_c')]:
        v=float(imu[src])
        if not np.isfinite(v) or v<=0:raise ValueError(f'Invalid IMU noise {src}')
        native['imu_params'][dst]=v
    if any(imu.get('a0',[0,0,0])) or any(imu.get('g0',[0,0,0])):raise ValueError('Nonzero input biases need explicit adaptation')
    native['imu_params'].update(g=float(imu['g']),imu_rate=float(imu['fps']),delay_imu_cam=0.)
    native['imu_initialization'].update(velocity=[0.,0.,0.],omega_bias=[0.,0.,0.],acc_bias=[0.,0.,0.])
    settings=copy.deepcopy(settings)
    if settings.get('runlc') or settings.get('use_ceres_backend') or not settings.get('use_imu') or not settings.get('pipeline_is_stereo'):
        raise ValueError('Requires author SchurVINS stereo-inertial without loop closure/Ceres backend')
    settings.update(calib_file=str(folder/'schurvins_calibration.yaml'),trace_dir=str(folder))
    return native,settings,T_BI

def imu_to_body(values,T_BI):
    result=values.copy();T_IB=np.linalg.inv(T_BI);rot=Rotation.from_quat(values[:,3:]).as_matrix()
    result[:,:3]+=np.einsum('nij,j->ni',rot,T_IB[:3,3]);result[:,3:]=Rotation.from_matrix(rot@T_IB[:3,:3]).as_quat()
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ['sequence_path','calibration_yaml','rgb_csv','exp_folder','settings_yaml']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exp_it',type=int,required=True);p.add_argument('--mode',choices=['stereo-vi'],default='stereo-vi');p.add_argument('--verbose',type=int,default=0)
    a=p.parse_args();f=a.exp_folder
    native,settings,TBI=native_settings(read_yaml(a.calibration_yaml),read_yaml(a.settings_yaml),f)
    nc=f/'schurvins_calibration.yaml';ns=f/'schurvins_native.yaml'
    for path,value in [(nc,native),(ns,settings)]:path.write_text(yaml.safe_dump(value,sort_keys=False))
    manifest=f/'schurvins_images.tsv';pairs=write_manifest(a.sequence_path,a.rgb_csv,manifest,['rgb_0','rgb_1'],.005)
    imu=a.sequence_path/'imu_0.csv'
    with imu.open() as stream:r=csv.reader(stream);next(r);origin=int(next(r)[0])-1_000_000_000
    raw=f/f'{a.exp_it:05d}_schurvins_imu.csv.tmp';binary=SOURCE/'build/vslamlab_schurvins_stereo_vi'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(repository='https://github.com/bytedance/SchurVINS',revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        binary_sha256=sha(binary),stereo_pairs=pairs,timestamp_origin_ns=origin,estimate_frame='body',
        pose_conversion='Native T_WI, T_WB=T_WI*inverse(T_BI)',status='running',
        input_sha256={str(p):sha(p) for p in [a.calibration_yaml,a.settings_yaml,nc,ns,manifest,imu]},patch_sha256={p.name:sha(p) for p in ADAPTER.glob('*.patch')})
    prov=f/f'schurvins_provenance_{a.exp_it:05d}.json';prov.write_text(json.dumps(record,indent=2))
    env=dict(os.environ)
    for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(key,None)
    rc=subprocess.run([str(binary),str(ns),str(manifest),str(imu),str(raw),str(origin)],env=env).returncode
    if rc:
        record.update(status='failed',returncode=rc);prov.write_text(json.dumps(record,indent=2));raise RuntimeError(f'SchurVINS exited {rc}; raw output retained')
    with raw.open() as stream:rows=list(csv.reader(stream))[1:]
    stamps=np.array([int(r[0]) for r in rows],np.int64);values=np.array([r[1:] for r in rows],float)
    if len(rows)<3 or np.any(np.diff(stamps)<=0) or not np.isfinite(values).all() or np.any(abs(np.linalg.norm(values[:,3:],axis=1)-1)>.005):raise RuntimeError('Invalid initialized SchurVINS trajectory')
    values=imu_to_body(values,TBI);target=f/f'{a.exp_it:05d}_KeyFrameTrajectory.csv';temp=target.with_suffix('.csv.tmp')
    with temp.open('w') as stream:w=csv.writer(stream);w.writerow(['ts (ns)','tx (m)','ty (m)','tz (m)','qx','qy','qz','qw']);w.writerows([int(t),*v] for t,v in zip(stamps,values))
    temp.replace(target);record.update(status='completed',initialized_poses=len(rows));prov.write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
