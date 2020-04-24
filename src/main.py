import numpy as np
import cv2
from util import *
from matplotlib import pyplot as plt
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--input-image", required=True, help="path to input image")
    ap.add_argument("-m", "--output-image", required=True, help="path to output image")
    ap.add_argument("-v", "--VP-image", required=True, help="path to VP image")
    args = vars(ap.parse_args())


    img = cv2.imread(args['input_image'])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1)
    Gmag_ = np.sqrt(sobelx**2.0 + sobely**2.0)
    Gdir_ = np.arctan2(sobely, sobelx)
    Gdir, idx = down_sample(Gmag_, Gdir_)


    beta = np.linspace(-np.pi/3, np.pi/3, 60)
    P = np.zeros_like(beta)

    for k in range(beta.shape[0]):
        a = 0.0
        b = beta[k]
        g = 0.0
        P[k] = pro_mixture(a, b, g, idx, Gdir)
    idx_coarse = np.argsort(P)
    b_opt = beta[idx_coarse[-1]]

    search_range = [-1.0, 0.0, 1.0]
    R_list = []
    P_list = np.zeros([len(search_range)**3,])
    c = 0

    for i in range(len(search_range)):
        b = b_opt + (np.pi)/90 * search_range[i]
        for j in range(len(search_range)):
            a = (np.pi)/36 * search_range[j]
            for k in range(len(search_range)):
                g = (np.pi) / 36 * search_range[k]
                R_list.append([a, b, g])
                P_list[c] = pro_mixture(a, b, g, idx, Gdir)
                c += 1
    idx_fine1 = np.argsort(P_list)
    R_opt = R_list[idx_fine1[-1]]


    search_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
    R_list = []
    P_list = np.zeros([len(search_range)**2,])
    c = 0
    b = R_opt[1]
    for i in range(len(search_range)):
        a = R_opt[0] + np.pi/36 * search_range[i]
        for j in range(len(search_range)):
            g = R_opt[2] + np.pi/36 * search_range[j]
            R_list.append([a, b, g])
            P_list[c] = pro_mixture(a, b, g, idx, Gdir)
            c += 1
    idx_fine2 = np.argsort(P_list)
    R_opt = R_list[idx_fine2[-1]]



    num_iter = 50
    R = angle2matrix(R_opt[0], R_opt[1], R_opt[2])
    S = matrix2vector(R)
    for i in range(num_iter):
        w_pm = E_step(S, idx, Gdir)
        opt = M_step(S, w_pm, idx, Gdir)
        S = opt.x
    R_em = vector2matrix(S)

    # plot pixels assignment to vanishing points
    im = plt.imshow(img)
    argmax_vp=np.argmax(w_pm,axis=1)
    for i in range(idx.shape[0]):
        if argmax_vp[i]==0:
            plt.scatter(y=idx[i,0]*5,x=idx[i,1]*5,color='red',s=1)
        elif argmax_vp[i]==1:
            plt.scatter(y=idx[i,0]*5,x=idx[i,1]*5,color='blue',s=1)
        elif argmax_vp[i]==2:
            plt.scatter(y=idx[i,0]*5,x=idx[i,1]*5,color='green',s=1)

    plt.savefig(args['output_image'], dpi=300)

    # plot vanishing point Locations
    im=plt.imshow(img)
    vp_loc=K.dot(R_em).dot(vp_dir)
    vp1=homo2img_coord(vp_loc[:,0])
    vp2=homo2img_coord(vp_loc[:,1])
    vp3=homo2img_coord(vp_loc[:,2])
    plt.scatter(y=vp3[1],x=vp3[0],color='green')
    plt.scatter(y=vp1[1],x=vp1[0],color='red')
    plt.savefig(args['VP_image'], dpi=300)