from AMCParser.amc_parser import parse_asf, parse_amc
import numpy as np
from AMCParser._3Dviewer import Viewer
import json

strict_walk_amc_file_lookup = {'02':['01','02'],'05':['01'],'06':['01'],'07':['01', '02', '03', '06', '07', '08', '09', '10', '11'],'08':['01', '02', '03', '06', '08', '09', '10'],'10':['04'],'12':['01','02','03'],'16':['15', '16', '21', '22', '31', '32', '47','58'],'35':['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12','13','14','15','16','28', '29', '30', '31', '32', '33','34'],'38':['01','02'],'39':['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '12','13','14'],'43':['01'],'49':['01'],'55':['04']}

body_parts = ['head', 'lclavicle', 'lfemur', 'lfingers', 'lfoot', 'lhand', 'lhipjoint', 'lhumerus', 'lowerback', 'lowerneck', 'lradius', 'lthumb', 'ltibia', 'ltoes', 'lwrist', 'rclavicle', 'rfemur', 'rfingers', 'rfoot', 'rhand', 'rhipjoint', 'rhumerus', 'root', 'rradius', 'rthumb', 'rtibia', 'rtoes', 'rwrist', 'thorax', 'upperback', 'upperneck']

used_body_parts = ['head', 'lclavicle', 'lfemur', 'lfingers', 'lfoot', 'lhand', 'lhumerus', 'lowerback', 'lowerneck', 'lradius', 'lthumb', 'ltibia', 'ltoes', 'lwrist', 'rclavicle', 'rfemur', 'rfingers', 'rfoot', 'rhand', 'rhumerus', 'root', 'rradius', 'rthumb', 'rtibia', 'rtoes', 'rwrist', 'thorax', 'upperback', 'upperneck']

body_dof_length = {'head': 3, 'lclavicle': 2, 'lfemur': 3, 'lfingers': 1, 'lfoot': 2, 'lhand': 2, 'lhumerus': 3, 'lowerback': 3, 'lowerneck': 3, 'lradius': 1, 'lthumb': 2, 'ltibia': 1, 'ltoes': 1, 'lwrist': 1, 'rclavicle': 2, 'rfemur': 3, 'rfingers': 1, 'rfoot': 2, 'rhand': 2, 'rhumerus': 3, 'root': 6, 'rradius': 1, 'rthumb': 2, 'rtibia': 1, 'rtoes': 1, 'rwrist': 1, 'thorax': 3, 'upperback': 3, 'upperneck': 3}

lower_body_parts = ['lfemur', 'ltibia', 'rfemur', 'rtibia']
total_frames = 0
all_frames = []

base_root = [  7.86289,  15.8386 , -40.2081 ,   9.03816,  -2.99522,  -3.21388]

fps = 15
downsampled_tracks = int(120/fps)
number_of_seconds = 2
seq_len = number_of_seconds*fps
for key,value in strict_walk_amc_file_lookup.items():
	asf_path = '/data/vikram/CS5335_Project/allasfamc/all_asfamc/subjects/'+key+'/'+key+'.asf'
	joints = parse_asf(asf_path)
	for v in strict_walk_amc_file_lookup[key]:
		amc_path = '/data/vikram/CS5335_Project/allasfamc/all_asfamc/subjects/'+key+'/'+key+'_'+v+'.amc'
		motions = parse_amc(amc_path)
		downsampled_motions = [[] for _ in range(downsampled_tracks)]
		for idx,m in enumerate(motions):
			sample_id = idx%downsampled_tracks
			s_motion_array = [motions[idx][k] for k in lower_body_parts]
			motion_array = []
			for s in s_motion_array:
				motion_array.extend(s)
			downsampled_motions[sample_id].append(motion_array)
		total_frames+=len(motions)
		for idx,d in enumerate(downsampled_motions):
			chop_off_id = len(d)%seq_len
			all_frames.extend(d[:-chop_off_id])




all_frames = np.array(all_frames)

# np.save('/home/vikram/git_folders/cs5335_project/TimeGAN-master/data/mocap_strict_walk_angles_lower_body_15fps_2sec_8angles.npy',all_frames)

# Visualise generated samples
import random

generated_data = []
for i in range(0, len(all_frames), seq_len):
	_x = all_frames[i:i + seq_len]
	generated_data.append(_x)
idx = np.random.permutation(len(generated_data))
ori_data = []
for i in range(len(generated_data)):
    ori_data.append(np.array(generated_data[idx[i]]))
asf_path = '02.asf'
joints = parse_asf(asf_path)
while 1:
	ch = random.choice(range(len(generated_data)))
	print(ch)
	seqs = generated_data[ch]
	motion_frames = []
	
	for idx,motion in enumerate(seqs):
		motion_frame = {}
		frame_idx = 0
		for k in lower_body_parts:
			motion_frame[k]=motion[frame_idx:frame_idx+body_dof_length[k]]
			frame_idx=frame_idx+body_dof_length[k]
		motion_frame['root']=base_root
		motion_frames.append(motion_frame)
	v = Viewer(joints, motion_frames)
	v.run()
