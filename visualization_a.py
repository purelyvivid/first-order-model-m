from demo import *
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence


def get_one_frame_in_gif_file(gif_pth, save_pth, i=20):
    im = Image.open(gif_pth)
    index = 0
    for frame in ImageSequence.Iterator(im):
        if index==i:
            frame.save(save_pth )
            print(f"Save file [{save_pth}]")
        index += 1
    

def make_animation_split_kp(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False, stn=None, oval_heatmap=False, kp_refiner=None, **kwargs):
    with torch.no_grad():
        predictions = []
        heatmaps = []
        masks = []
        sparse_deformeds = []
        kp_driving_list = []
        kp_source_list = []
        kp_norm_list = []
        
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        #print(kp_source['value'].shape, kp_driving_initial['value'].shape)
        # --- add start ---
        adj_source, adj_source, adj_weights_source = None, None, None
        if not kp_refiner is None:
            #print("use kp_refiner..")
            kp_source['value'], adj_source, adj_weights_source = kp_refiner(kp_source['value'], **kwargs)
            kp_driving_initial['value'], _, _ = kp_refiner(kp_driving_initial['value'], **kwargs)
        # --- add end ---
        
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            
            # --- add start ---
            adj_driving, adj_driving, adj_weights_driving= None, None, None
            if not kp_refiner is None:
                #print("use kp_refiner..")
                kp_driving['value'], adj_driving, adj_weights_driving = kp_refiner(kp_driving['value'], **kwargs)
            # --- add end ---
            
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            sparse_deformeds.append(out['sparse_deformed'])
            masks.append(out['mask'])
            kp_driving_list.append(kp_driving)
            kp_source_list.append(kp_source)
            kp_norm_list.append(kp_norm)
            heatmaps.append(out['heatmap_representation'])

    return predictions, masks, heatmaps, sparse_deformeds, kp_source_list, kp_driving_list, kp_norm_list, adj_source, adj_driving, adj_weights_source, adj_weights_driving



tensorimg_to_nparr = lambda x : np.array(x.permute(1, 2, 0))
resize_256_to_64 = lambda x : np.array(Image.fromarray( (x*255).astype("uint8") ).resize((64,64)))/255 
resize_64_to_256 = lambda x : np.array(Image.fromarray( (x*255).astype("uint8") ).resize((256,256)))/255 
nparrimg_to_tensor  = lambda x : torch.tensor(x).permute(2, 0, 1)
tensor_to_nparrimg  = lambda x : x.permute(1, 2, 0).cpu().numpy()
rescale_1_to_255 = lambda x : np.array(np.clip(x*255, 0, 255).astype("uint8"))


def show_img(img, save_pth="", save=False):# tensor_image: [3, H, W]
    plt.imshow(img)
    plt.axis('off')
    if save:
        plt.savefig(save_pth, dpi=1000)
    plt.show()
    
def draw_heatmap_single_frame(predictions, source_image, driving_video, masks, heatmaps, sparse_deformeds, kp_source_list, kp_driving_list, kp_norm_list, save_pth, i=2):
    collect = []
    for j in range(sparse_deformeds[i].shape[1]):#對每一個 kp

        deformed = sparse_deformeds[i][0][j]
        driving = nparrimg_to_tensor(resize_256_to_64(driving_video[i]))
        prediction = nparrimg_to_tensor(resize_256_to_64(predictions[i]))

        mask = torch.stack( [masks[i][0][j]]*3, 0)
        mask = (mask/2+0.5)
        img_de_m = mask*deformed
        img_de_m /= img_de_m.max() # deformed with mask

        img_dr_m = mask*driving
        img_dr_m /= img_dr_m.max()  # driving with mask  

        img_pr_m = mask*prediction
        img_pr_m /= img_pr_m.max()  # prediction with mask

        heatmap_ = heatmaps[i][0][j]
        dummy = torch.zeros_like(heatmap_)
        heatmap = torch.cat( [heatmap_]+[dummy]*2, 0) 
        #print(heatmap.max(), heatmap.min())
        heatmap = (np.clip(heatmap,-1,1)+1)/0.5
        img_de_h = heatmap*deformed
        img_de_h /= img_de_h.max()  # deformed with heatmap

        img_dr_h = heatmap*driving
        img_dr_h /= img_dr_h.max()  # driving with heatmap  

        img_pr_h = heatmap*prediction
        img_pr_h /= img_pr_h.max()  # prediction with heatmap

        imgs = [  img_de_m, img_dr_m, img_pr_m, img_de_h, img_dr_h, img_pr_h ] 
        concated = np.concatenate([ tensorimg_to_nparr(img)  for img in imgs], 1)
        #show_img( concated , save_pth=save_pth+f"kp[{j}].png", save=True)
        
        collect .append(concated)
    collect = np.concatenate(collect,0)
    print(collect.shape)
    imageio.imsave(save_pth+f"kps.png", collect)
    return collect 
        

def draw_heatmap_all_frame(driving_video, predictions, masks, heatmaps, \
                           sparse_deformeds, kp_source_list, kp_driving_list, \
                           kp_norm_list, save_pth, fps):  
    # 256 to 64
    for j in range(sparse_deformeds[0].shape[1]):#對每一個 kp
        print("kp:", j)
        predictions_ = [] 
        for i in range(len(predictions)): #對每一個 frame

            deformed = sparse_deformeds[i][0][j]
            driving = nparrimg_to_tensor(resize_256_to_64(driving_video[i]))
            prediction = nparrimg_to_tensor(resize_256_to_64(predictions[i]))

            mask = torch.stack( [masks[i][0][j]]*3, 0)
            mask = (mask/2+0.5)
            img_de_m = mask*deformed
            img_de_m /= img_de_m.max() # deformed with mask

            img_dr_m = mask*driving
            img_dr_m /= img_dr_m.max()  # driving with mask  

            img_pr_m = mask*prediction
            img_pr_m /= img_pr_m.max()  # prediction with mask

            heatmap_ = heatmaps[i][0][j]
            dummy = torch.zeros_like(heatmap_)
            heatmap = torch.cat( [heatmap_]+[dummy]*2, 0) 
            #print(heatmap.max(), heatmap.min())
            heatmap = (np.clip(heatmap,-1,1)+1)/0.5
            img_de_h = heatmap*deformed
            img_de_h /= img_de_h.max()  # deformed with heatmap

            img_dr_h = heatmap*driving
            img_dr_h /= img_dr_h.max()  # driving with heatmap  

            img_pr_h = heatmap*prediction
            img_pr_h /= img_pr_h.max()  # prediction with heatmap

            imgs = [ img_de_m, img_dr_m, img_pr_m, img_de_h, img_dr_h, img_pr_h ] 
            concated = np.concatenate([ tensorimg_to_nparr(img)  for img in imgs], 1)
            #show_img( concated )
            predictions_.append( concated )

        imageio.mimsave(save_pth+f"kp[{j}].gif", [img_as_ubyte(frame) for frame in predictions_], fps=fps)

"""     
def draw_graph_on_fig(imgarr, node_coords, edges):
    dpi = imgarr.shape[0] 
    w,h = 1,1
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for kp in node_coords:
        ax.scatter(kp[0],kp[1], c="r")
    for (es,en) in edges:
        ax.plot(node_coords[es], node_coords[en], c="r")
    fig.savefig("tmp.png", dpi=dpi)
    return np.array(Image.open("tmp.png"))[:,:,:3]

"""   