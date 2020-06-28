# Unsupervised Key Point Learning with Graph

## Motivation

Auther: Hui-Yu Huang (purelyvivid@gmail.com, NCTU) 2020/06

First-Order-Model[1] (NIPS 19), an advanced image animation approach by unsupervised key point detection, has shown extraordinary capability in sparse motion transformation. However, number of key points is a preset hyperparameter and key points learn independently on each heatmaps.

To leverage the relation between key points, I propose a method which regards key points as a **Graph**, modeled by Attention-Based Graph Neural Network. **Graph constraints the key points and shows how key points related to each other, providing informative prior for model training, and can be visualized for human understanding.**

## Proposed Model
![](https://i.imgur.com/fHsLCm5.png)

## Result
- **Original Model**
    - Left to right: ( source image - driving video - predictions - driving video with Key Point - predictions with Key Point )
    
![](demo/202006271600/_kps_result_sc00076_dr00075_m0.gif)


- **Proposed Model**
    - Left to right: ( source image - driving video - predictions - driving video with Key Point Graph - predictions with Key Point Graph )
    
![](demo/202006271600/_kps_result_sc00076_dr00075_m1_gw.gif)



## Visualization
- **Original Model**
    - Find the most motion frame: 04
    - Left to right: ( source image - driving video - predictions - driving video with Heatmap and Key Point - predictions with with Heatmap and Key Point )
    - Up to down: (key point NO. : background + NO. 1~10 ) 
    
    
<img src="demo/202006271600/frame[04]_sc00076_dr00075_m0.png" width="600"/>


- **Proposed Model**
    - Find the most motion frame: 10
    - Left to right: ( source image - driving video - predictions - driving video with Heatmap and Key Point Graph - predictions with with Heatmap and Key Point Graph )
    - Up to down: (key point NO. : background + NO. 1~10 ) 
    
    
<img src="demo/202006271600/frame[10]_sc00076_dr00075_m1_gw.png" width="600"/>




## Reference

[1] This repository is modefied from the [source code](https://github.com/AliaksandrSiarohin/first-order-model/) for the paper [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) 
