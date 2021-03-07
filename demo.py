import argparse
import numpy as np
from pathlib import Path
import math
import matlab.engine
import os
import cv2
from model_unet import get_model as get_sub_A
from model_unet_stepA import get_model as get_sub_B
from keras import backend as K
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_coded", type=str, default= 'balldrill_832x480_QP42_8F.yuv', required=True,  help="test compressed video")
    parser.add_argument("--video_ori", type=str, default= 'balldrill_832x480_ori_8F.yuv', required=True, help="test compressed video")
    parser.add_argument("--QP", type=int, default=42, required=True, help="QP Value")
    parser.add_argument("--output_dir", type=str, default='output_A', help="if set, save resulting sequence")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    return args

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

def sub_B_process(model, im):
    layer_X = K.function([model.layers[0].input], [model.layers[35].output])

    depth = 5    
    num_input = 3
    h, w, _ = im.shape
            
    n_rows = math.floor(h/(2**depth))*(2**depth)
    n_cols = math.floor(w/(2**depth))*(2**depth)
    
    szh = h//2;
    szw = w//2
    xim = np.zeros((szh, szw, num_input), dtype=np.uint8)
    xim = cv2.resize(im[:n_rows, :n_cols, :], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    xim = np.expand_dims(xim.astype(np.float), axis=0)
    fmap = layer_X([xim])[0]

    return fmap


def main():
    args = get_args()
    rows = args.height
    cols = args.width

    depth = 5  
    n_rows = math.floor(rows/(2**depth))*(2**depth)
    n_cols = math.floor(cols/(2**depth))*(2**depth)    

    video_path = args.video_coded
    ori_path   = args.video_ori
    weight_sub_B = 'models/QP' + str(args.QP) + '-HALF-XH.hdf5'
    weight_sub_A = 'models/QP' + str(args.QP) + '-FINAL.hdf5'
    print(weight_sub_A)
    model_A = get_sub_A('unet')
    model_B = get_sub_B('unet')
    model_A.load_weights(weight_sub_A)
    model_B.load_weights(weight_sub_B)
    num_input = 3
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(video_path) == False:
        os._exit(0)

    len = os.path.getsize(video_path)
    num_F = len//(rows*cols*3//2)
    out_image = np.zeros((rows, cols * 2, 1), dtype=np.uint8)
    
    eng = matlab.engine.start_matlab()

    for n in range(0, num_F):        
        [comp_im, skip_F] = eng.frame_compensation(video_path, n+1, num_F, rows, cols, nargout=2)  
        ori_im = eng.read_y(ori_path, n+1, rows, cols, nargout=1)
        comp_im = np.array(comp_im)
        ori_data = np.array(ori_im)
        
        if(skip_F == 1):
            out_image = np.zeros((n_rows, n_cols*3, 1), dtype=np.uint8)
        
            out_image[:, :n_cols, 0] = ori_data[:n_rows, :n_cols]
            out_image[:, n_cols:n_cols*2, 0] = comp_im[:n_rows, :n_cols, 0]
            out_image[:, n_cols*2:n_cols*3, 0] = comp_im[:n_rows, :n_cols, 1]

            str_name = '%(name)05d.png'%{'name':n}
            cv2.imwrite(str(output_dir.joinpath(str_name)), out_image)            

            continue

        xim = comp_im[:n_rows, :n_cols, :]
        ori_data= ori_data[:n_rows, :n_cols]

       fmap = sub_B_process(model_B, xim)
        xim = np.expand_dims(xim.astype(np.float), 0)
        pred = model_A.predict([xim, fmap]) 
        rim  = get_image(pred[0])       

        out_image = np.zeros((n_rows, n_cols*3, 1), dtype=np.uint8)
        
        out_image[:, :n_cols, 0] = ori_data[:n_rows, :n_cols]
        out_image[:, n_cols:n_cols*2, 0] = comp_im[:n_rows, :n_cols, 0]
        out_image[:, n_cols*2:n_cols*3, 0] = rim[:n_rows, :n_cols, 0]

        str_name = '%(name)05d.png'%{'name':n}
        cv2.imwrite(str(output_dir.joinpath(str_name)), out_image)   
        print('Frame:%4d'%(n))         

if __name__ == '__main__':
    main()