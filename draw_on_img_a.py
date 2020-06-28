import sympy
from scipy.spatial import ConvexHull
import numpy as np
from skimage.draw import polygon, circle

def verify_dist(p_true, p_pred, dist):
    return (( (p_true[0]-p_pred[1])**2 - (p_true[0]-p_pred[1])**2)**0.5 - dist)<1e-10


def line2rectangle(p1, p2, linewidth=1):
    p1x, p1y, p2x, p2y = map(float, [p1[0],p1[1], p2[0], p2[1]] )   
        
    a = (p2y-p1y)/(p2x-p1x)
    """
    p1, p2 通過 y = a x + b ，
    與 y = a x + b 垂直的線是： 
    y = (-1/a) x + c ，p1 與 p2 通過此線
    => c = y - (-1/a) x
    """
    rev_a = 0 if a==0 else (-1/a)
    c1 = p1y - rev_a*p1x
    c2 = p2y - rev_a*p2x
    """
    假設 x,y 在 y = (-1/a) * x + c1 上, 因為 p1 也在此線上, 跟 p1 距離為 linewidth 的公式為
    (x-p1x)**2 + (y-p1y)**2 = linewidth**2
    則 (x-p1x)**2 + ((-1/a)*x + c1 - p1y)**2 = linewidth**2
    """
    formula = "(x-{})**2 + ({}*x+{}-{})**2 - {}**2"
    sol1 = sympy.solve(formula.format(p1x,rev_a,c1,p1y,linewidth ) )
    sol2 = sympy.solve(formula.format(p2x,rev_a,c2,p2y,linewidth ) )
    assert len(sol1)==2 and len(sol2 )==2
    x11, x12 = sol1
    x21, x22 = sol2
    y11, y12 = rev_a * x11 + c1, rev_a * x12 + c1
    y21, y22 = rev_a * x21 + c2, rev_a * x22 + c2
    p11 = (x11, y11)
    p12 = (x12, y12)
    p21 = (x21, y21)
    p22 = (x22, y22)
    
    # 驗證
    b = True
    b &= verify_dist(p1, p11, linewidth) 
    b &= verify_dist(p1, p12, linewidth) 
    b &= verify_dist(p2, p21, linewidth) 
    b &= verify_dist(p2, p22, linewidth) 
    
    if not b:
        print("verify_dist error !!")
        
    return np.array([p11, p12, p21, p22])



def get_convex_points(points):
    hull = ConvexHull(points)
    points = points[hull.vertices,:]
    return points

def idx_filter(rr, cc, dpi):
    cond = (rr>=0)&(rr<dpi)&(cc>=0)&(cc<dpi)
    rr, cc = rr[cond] , cc[cond]
    return rr, cc

def exp(x):
    """
    >>> exp(0.1), exp(1), exp(1.7)
    (0.5525854590378239, 1.3591409142295225, 2.7369736958636)
    """
    return np.exp(x)/2

def rescale(x, xs):
    return (x-xs.min())/(xs.max()-xs.min())
      

def draw_graph_on_fig(imgarr, node_coords, edges, ehn_idx=-1, edge_weights=None):
        
    imgarr = imgarr.copy()
    dpi = imgarr.shape[0]
     
    # clip node coord
    chg_coord = lambda x : int((np.clip(x, -1., 1.)+1)/2*(dpi-1))
    for i, kp in enumerate(node_coords):
        node_coords[i][0] = chg_coord(kp[0])
        node_coords[i][1] = chg_coord(kp[1])
    #node_coords = node_coords.astype("uint8")
        
    #print(node_coords)
    
    # draw edges
    linewidth = 1
    for edge_i, (idx_s,idx_n) in enumerate(edges):
        p1, p2 = node_coords[idx_s], node_coords[idx_n]
        if (p1 != p2).all(): # x,y 任何一個相等都不能接受
            w = edge_weights[edge_i] if (not edge_weights is None) else 0.0
            if w>0.0:
                linewidth_ = linewidth*exp(w) if (not edge_weights is None) else linewidth
                #linewidth_ =  linewidth
                points = line2rectangle(p1, p2, linewidth=linewidth_) 
                points = get_convex_points(points)
                r = points[:,0]
                c = points[:,1]
                rr, cc = polygon(r, c)
                rr, cc = idx_filter(rr, cc, dpi) 
                rsw =  rescale(w, edge_weights)
                imgarr[rr, cc, 1] = 0    # G channels
                imgarr[rr, cc, 2] = 1-rsw    # B channels
                imgarr[rr, cc, 0] = rsw   # R channels
            
    # draw nodes
    radius = 2 
    for idx, kp in enumerate(node_coords): 
        if idx==0: continue #第 0 個kp是背景 忽略
        rds = radius*1.5 if ehn_idx==idx else radius
        rr, cc  = circle(kp[0], kp[1], rds)
        rr, cc = idx_filter(rr, cc, dpi)
        if ehn_idx==idx:
            imgarr[rr, cc, 0] = 1  # R channels if it's ehn_idx
            imgarr[rr, cc, 1:] = 0
        else:
            imgarr[rr, cc, 1] = 1   # G channels if it's normal idx
            imgarr[rr, cc, 0] = 0 
            imgarr[rr, cc, 2] = 0 
            
             
    return imgarr