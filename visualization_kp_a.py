from demo import *
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

from draw_on_img_a import draw_graph_on_fig

tensor_to_nparr  = lambda x : x.cpu().numpy()
nparr_to_tensor  = lambda x : torch.tensor(x)

tensorimg_to_nparrimg = lambda x : tensor_to_nparr(x.permute(1, 2, 0))
nparrimg_to_tensorimg  = lambda x : nparr_to_tensor(x).permute(2, 0, 1)

rescale_1_to_255 = lambda x : np.array(np.clip(x*255, 0, 255).astype("uint8"))
rescale_255_to_1 = lambda x : np.array(np.clip(x/255, 0, 1).astype("float64"))

resize = lambda nparr, size : np.array(Image.fromarray( nparr.astype("uint8") ).resize(size))
nparrimg_resize_to_64 = lambda nparr : resize(nparr, (64,64))
nparrimg_resize_to_256 = lambda nparr : resize(nparr, (256,256))

def _gaussian2kp(heatmap, kp_detector):#heatmap (h,w)
    kp_x, kp_y = kp_detector.gaussian2kp(heatmap.unsqueeze(0).unsqueeze(0))['value'][0][0]
    return torch.tensor((kp_y, kp_x)) #kp['value'] (2,)
    
def heatmap2gaussian(heatmap):#heatmap (64,64)
    #print("0 max, min", heatmap.max(),heatmap.min())
    if heatmap.max()-heatmap.min()>0:
        
        heatmap_driving = heatmap.clone()
        heatmap_source = heatmap.clone()
        heatmap_source = - heatmap_source

        heatmap_driving[heatmap_driving<0] = 0
        heatmap_source[heatmap_source<0] = 0
        
        heatmap_driving /= heatmap_driving.sum()
        heatmap_source /= heatmap_source.sum()
    else:
        heatmap_driving = heatmap.clone()
        heatmap_source = heatmap.clone()
    #print("heatmap_driving.shape",heatmap_driving.shape)
    return heatmap_driving, heatmap_source 


def transform(tensor64_1channel_0to1):
    tensor64_3channel_0to1 = torch.stack([tensor64_1channel_0to1]*3,0)  #; print("tensor64_3channel_0to1",tensor64_3channel_0to1.shape)
    nparr64_3channel_0to1 = tensorimg_to_nparrimg(tensor64_3channel_0to1)
    nparr64_3channel_0to255 = rescale_1_to_255(nparr64_3channel_0to1)
    nparr256_3channel_0to255 = nparrimg_resize_to_256(nparr64_3channel_0to255)
    nparr256_3channel_0to1 = rescale_255_to_1(nparr256_3channel_0to255 )
    tensor256_3channel_0to1 = nparrimg_to_tensorimg(nparr256_3channel_0to1)
    tensor256_1channel_0to1 = tensor256_3channel_0to1[0,:,:]#  (256,256) , val [0,1] (sum is 1)
    if tensor256_1channel_0to1.max() != tensor256_1channel_0to1.min():
        tensor256_1channel_0to1 /= tensor256_1channel_0to1.max()#  (256,256) , val [0,1] (max/max is 1)
    return tensor256_1channel_0to1

def transform_1(tensor256_1channel_0to1):
    tensor256_1channel_0p5to1 = (torch.clamp(tensor256_1channel_0to1,0.,1.)+1)*0.5 # (256,256) , val [0.5,1]  # be gray an transperant
    tensor256_3channel_0p5to1 =  torch.stack( [tensor256_1channel_0p5to1]*3, 2)  # (256,256,3) , val [0.5,1]    
    return tensor256_3channel_0p5to1

tensor_draw_graph_on_fig = lambda x, V, E, ehn_idx, edge_weights : \
    nparr_to_tensor(draw_graph_on_fig(tensor_to_nparr(x), V, E, ehn_idx=ehn_idx, edge_weights=edge_weights))
kp_dict_to_nparr = lambda kp_dict : np.array(kp_dict['value'][0])

def adj_to_edges(adj):
    edges = np.where(adj)
    weights = adj[edges]    
    return np.stack(edges).T, np.stack(weights).T

def draw_heatmap_all_frame_w_graph_on_fig(kp_detector, save_name, source_image, driving_video, predictions, masks, heatmaps, \
                           sparse_deformeds, kp_source_list, kp_driving_list, \
                           kp_norm_list, adj_source, adj_driving, adj_weights_source, adj_weights_driving,\
                           save_pth, fps, graph_with_weight=False, gen_kp_gif=True):  
    
    # 64 to 256
    """
    info[frame_no][item][kp_no]
    """
    if adj_weights_source is None or adj_weights_driving is None:
        if graph_with_weight == True: 
            graph_with_weight = False 
            print("chg graph_with_weight = False ")
    gw = "_gw" if graph_with_weight else ""
    if not None in [adj_source, adj_driving, adj_weights_source, adj_weights_driving]:
        adj_source_ =  adj_weights_source if graph_with_weight else adj_source
        adj_driving_ = adj_weights_driving if graph_with_weight else adj_driving

        edge_source, edge_weights_source = adj_to_edges(adj_source_) 
        edge_driving, edge_weights_driving = adj_to_edges(adj_driving_) 

        edge_source = edge_source +1
        edge_driving = edge_driving +1
    else:
        edge_source, edge_weights_source = [], []
        edge_driving, edge_weights_driving = [], []         

    source_image = nparr_to_tensor(source_image)
    
    
    num_frame = len(predictions)
    num_kp = sparse_deformeds[0].shape[1]
    info = {}
    for i in range(num_frame): #對每一個 frame
        driving_video_ = nparr_to_tensor(driving_video[i])  # (256,256,3), val [0,1]
        prediction = nparr_to_tensor(predictions[i]) # (256,256,3), val [0,1]
        #進行 儲存 dict 的初始化
        info[i] = {
            'driving_video': driving_video_,
            'prediction' : prediction,
            'mask' : [],
            'heatmap_driving' : [],
            'heatmap_source' : [],
            'heatmap_diff' : None,
            'kp_driving' : None,
            'kp_source' : None,
        }
             
        kp_driving_ = []
        kp_source_ = []
        kp_norm_ = []
        heatmap_diff = []
        for j in range(num_kp):#對每一個 kp
            

            mask = masks[i][0][j] # (64,64) , val [0,1]
            #mask = (mask/2+0.5)   # be gray an transperant, val [0.5,1.0]
            mask = transform(mask) # (64,64) to (256,256), val [0.5,1.0]
            info[i]['mask'].append(mask)

            heatmap_ = heatmaps[i][0][j][0] # (64,64) , val [-1,+1]  
            heatmap_diff.append(heatmap_.max()-heatmap_.min())
            heatmap_driving, heatmap_source  = heatmap2gaussian(heatmap_)# (64,64) val [0,1] 
            
            kp_driving = _gaussian2kp(heatmap_driving, kp_detector) # (64,64) val [0,1] to (2,)
            kp_source = _gaussian2kp(heatmap_source, kp_detector) # (64,64) val [0,1] to (2,)
            kp_driving_.append(kp_driving.numpy())
            kp_source_.append(kp_source.numpy())
            #print(i,j,kp_driving_[-1], kp_source_[-1])
            
            heatmap_driving = transform(heatmap_driving) # (64,64) val [0,1] to (256,256) , val [0,1] 
            heatmap_source = transform(heatmap_source) # (64,64) val [0,1] to (256,256) , val [0,1] 
            heatmap_driving_ = transform_1(heatmap_driving)# (256,256) , val [0,1] to (256,256,3) , val [0.5,1] 
            heatmap_source_ = transform_1(heatmap_source)# (256,256) , val [0,1] to (256,256,3) , val [0.5,1]
            info[i]['heatmap_driving'].append(heatmap_driving_)
            info[i]['heatmap_source'].append(heatmap_source_) 
        
        info[i]['kp_driving'] = np.stack(kp_driving_, 0)
        info[i]['kp_source'] =  np.stack(kp_source_, 0)
        info[i]['heatmap_diff'] = np.mean(heatmap_diff)
    
    #對每一個 frame  
    best_frame_idx = np.argmax([info[i]['heatmap_diff'] for i in range(num_frame)]) #find best frame
    print("best frame:", best_frame_idx)
        
    for i in range(num_frame): #對每一個 frame  
        if i != best_frame_idx : continue #find best frame
        out = []
        for j in range(num_kp):#對每一個 kp  
            img_dr = info[i]['driving_video'].clone()
            img_pr = info[i]['prediction'].clone()
            hm_dr = info[i]['heatmap_driving'][j].clone()
            hm_sc = info[i]['heatmap_source'][j].clone()
            kp_dr = info[i]['kp_driving'].copy()
            kp_sc = info[i]['kp_source'].copy()
            img_pr_g = tensor_draw_graph_on_fig(img_pr*hm_dr, kp_dr, edge_driving, ehn_idx=j, edge_weights=edge_weights_driving if graph_with_weight else None)
            img_dr_g = tensor_draw_graph_on_fig(img_dr*hm_sc, kp_sc, edge_source, ehn_idx=j, edge_weights=edge_weights_source if graph_with_weight else None)
            #print(img_dr_g.shape)
            imgs = [ source_image, img_dr, img_pr, img_dr_g, img_pr_g ] 
            concated = np.concatenate([ tensor_to_nparr(img)  for img in imgs], 1) 
            out.append(concated)
            #print(concated.shape)
        out = np.concatenate([ img for img in out], 0) 
        #print(out.shape)
        sn = save_pth+f"frame[{str(i).zfill(2)}]{save_name}{gw}.png"
        imageio.imsave(sn, out)
        print(f"save [{sn}]!")
           
    """
    Animation: gen_kp_gif
    """  
    if gen_kp_gif :
        predictions_ = []
        for i in range(num_frame): #對每一個 frame   
            img_dr = info[i]['driving_video'].clone()
            img_pr = info[i]['prediction'].clone()
            kp_dr = info[i]['kp_driving'].copy()
            kp_sc = info[i]['kp_source'].copy()            
            img_pr_g = tensor_draw_graph_on_fig(img_pr, kp_dr, edge_driving, ehn_idx=-1, edge_weights=edge_weights_driving if graph_with_weight else None)
            img_dr_g = tensor_draw_graph_on_fig(img_dr, kp_sc, edge_source, ehn_idx=-1, edge_weights=edge_weights_source if graph_with_weight else None)
            imgs = [ source_image, img_dr, img_pr, img_dr_g, img_pr_g ] 
            concated = np.concatenate([ tensor_to_nparr(img)  for img in imgs], 1) 
            #show_img( concated )
            predictions_.append( concated )

        sn = save_pth+f"kps_result{save_name}{gw}.gif"
        imageio.mimsave(sn, [img_as_ubyte(frame) for frame in predictions_], fps=fps)
        print(f"save [{sn}]!")