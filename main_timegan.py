"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from AMCParser.amc_parser import parse_asf, parse_amc
from AMCParser._3Dviewer import Viewer

body_dof_length = {'head': 3, 'lclavicle': 2, 'lfemur': 3, 'lfingers': 1, 'lfoot': 2, 'lhand': 2, 'lhumerus': 3, 'lowerback': 3, 'lowerneck': 3, 'lradius': 1, 'lthumb': 2, 'ltibia': 1, 'ltoes': 1, 'lwrist': 1, 'rclavicle': 2, 'rfemur': 3, 'rfingers': 1, 'rfoot': 2, 'rhand': 2, 'rhumerus': 3, 'root': 6, 'rradius': 1, 'rthumb': 2, 'rtibia': 1, 'rtoes': 1, 'rwrist': 1, 'thorax': 3, 'upperback': 3, 'upperneck': 3}

lower_body_parts = ['lfemur',  'ltibia', 'rfemur', 'rtibia']


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.data_name in ['stock', 'energy','mocap','mocap_strict']:
    temp_ori_data,min_val,max_val = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  
  idx = np.random.permutation(len(temp_ori_data))    
  ori_data = []
  for i in range(len(temp_ori_data)):
    ori_data.append(np.array(temp_ori_data[idx[i]]))
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  parameters['seq_len'] = args.seq_len
  
  generated_data, ori_data = timegan(ori_data, parameters,min_val,max_val)   
  print('Finish Synthetic Data Generation')
  # print(generated_data)
  print(generated_data.shape)
  # Performance metrics   
  # Output initialization

  
  visualization(ori_data, generated_data, 'pca')
  visualization(ori_data, generated_data, 'tsne')
  


  return ori_data, generated_data


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy','mocap','mocap_strict'],
      default='stock',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data = main(args)
  asf_path = '02.asf'
  joints = parse_asf(asf_path)
  base_root = [  7.86289,  15.8386 , -40.2081 ,   9.03816,  -2.99522,  -3.21388]


  while 1:
    ch = random.choice(range(generated_data.shape[0]))
    print(ch)

    seqs = generated_data[ch]
    motion_frames = []
    for motion in seqs:
      motion_frame = {}
      frame_idx = 0
      for k in lower_body_parts:
        motion_frame[k]=motion[frame_idx:frame_idx+body_dof_length[k]]
        frame_idx=frame_idx+body_dof_length[k]

      motion_frame['root']=base_root
      motion_frames.append(motion_frame)

    v = Viewer(joints, motion_frames)
    v.run()