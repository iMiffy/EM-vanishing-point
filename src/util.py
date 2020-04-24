import numpy as np
import scipy.io
from scipy.optimize import least_squares
import scipy.stats

cam_data = scipy.io.loadmat('cameraParameters.mat')
f = cam_data['focal']
pixelSize = cam_data['pixelSize']
pp = cam_data['pp']
K = np.array([[f[0][0]/pixelSize[0][0], 0, pp[0][0]], [ 0 ,f[0][0]/pixelSize[0][0], pp[0][1]],  [0, 0, 1]], dtype=np.float32)
vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)

P_m_prior = [0.13, 0.24, 0.13, 0.5]
sig = 0.5
mu = 0.0


def remove_polarity(x):
    '''
    :param x:  the angle differences between the predicted normal direction and the gradient direction of a pixel.
               x is in shape [3,] which represent the normal direction with respect to the three edge models.
    :return: the minimal value after add pi and -pi
    '''
    x = np.expand_dims(x, axis=0)
    new = np.abs(np.concatenate([x, x + np.pi, x - np.pi], axis= 0))
    diff = np.min(new, axis=0)
    return diff

def homo2img_coord(x):
    u=x[0]/x[2]
    v=x[1]/x[2]
    return np.array([u,v])

def angle2matrix(a, b, g):
    '''

    :param a: the rotation angle around z axis
    :param b: the rotation angle around y axis
    :param g: the rotation angle around x axis
    :return: rotation matrix
    '''

    R = np.array([[np.cos(a)*np.cos(b), -np.sin(a)*np.cos(g)+np.cos(a)*np.sin(b)*np.sin(g),
                   np.sin(a)*np.sin(g)+np.cos(a)*np.sin(b)*np.cos(g), 0],
    [np.sin(a)*np.cos(b),  np.cos(a)*np.cos(g)+np.sin(a)*np.sin(b)*np.sin(g),
     -np.cos(a)*np.sin(g)+np.sin(a)*np.sin(b)*np.cos(g), 0],
     [-np.sin(b) , -np.cos(b)*np.sin(g), np.cos(b)*np.cos(g), 0]],
     dtype=np.float32)

    return R

def vector2matrix(S):

    '''
    :param S: the Cayley-Gibbs-Rodrigu representation
    :return: rotation matrix R
    '''
    S = np.expand_dims(S, axis=1)
    den = 1 + np.dot(S.T, S)
    num = (1 - np.dot(S.T, S))*(np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
    R = num/den
    homo = np.zeros([3,1], dtype=np.float32)
    R = np.hstack([R, homo])
    return R

def skew(a):
    s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
    return s

def matrix2quaternion(T):

    R = T[:3, :3]

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def matrix2vector(R):
    '''
    :param R: the camera rotation marix
    :return:  the Cayley-Gibbs-Rodrigu representation
    '''
    Q = matrix2quaternion(R)
    S = Q[1:]/Q[0]
    return S


def vp2dir(K, R, u):
    '''
    :param K: camera intrinsic matrix
    :param R: camera rotation matrix
    :param u: pixel location represented in homogeneous coordinate [x, y, 1]
    :return: the estimated normal direction for edge that pass through pixel u
    '''
    vp_trans = K.dot(R).dot(vp_dir)
    vp_trans = K.dot(R).dot(vp_dir)
    edges = np.cross(vp_trans.transpose(), u)
    thetas_es = np.arctan2(edges[:, 1], edges[:, 0])
    return thetas_es

def down_sample(Gmag_, Gdir_):
    '''
    :param Gmag_: gradient magtitude of the original image
    :param Gir_: gradient direction of the original image
    :return: pixels we will use in the EM algorithm and the corresponding gradient direction
    '''
    Gmag = Gmag_[4::5, 4::5]
    Gdir = Gdir_[4::5, 4::5]
    threshold = np.sort(np.reshape(Gmag, [Gmag.shape[0]*Gmag.shape[1]]))
    idx = np.argwhere(Gmag > threshold[-2001])
    return Gdir, idx

def pro_mixture(a, b, g, idx, Gdir):
    '''
    :param a, b, g_: camera rotation parameters
    :return: p_image
    '''

    R = angle2matrix(a, b, g) # convert the angles into rotation matrix
    p_image = 0.0   # initialise posterior setting to zero


    P_ang=np.zeros(shape=(idx.shape[0],4))
    for i in range(idx.shape[0]):
        # pixel Location
        p=np.array([idx[i,1]*5+4,idx[i,0]*5+4,1],dtype=float)
        # theta and phi (pixel gradient)
        Theta=vp2dir(K, R, p)
        phi=Gdir[idx[i,0],idx[i,1]]
        error=remove_polarity(phi-Theta)
        # P ang
        P_ang[i,:3]=np.nan_to_num(scipy.stats.norm(mu, sig).pdf(error))
        P_ang[i,3]=1/(2.0*np.pi)
        # evidence
    p_image = np.sum(np.nan_to_num(np.log(P_ang.dot(np.array(P_m_prior)))))


    return p_image

def E_step(S, idx, Gdir):
    '''
    :param S : the Cayley-Gibbs-Rodrigu representation of camera rotation parameters
    :return: w_pm
    '''
    R = vector2matrix(S)  # Note that the 'S' is just for optimization, it has to be converted to R during computation
    w_pm = np.zeros([idx.shape[0], 4], dtype=np.float32)

    P_ang=np.zeros(shape=(idx.shape[0],4))
    # to be implemented, the E-step to compute the weights for each vanishing point at each pixel
    for i in range(idx.shape[0]):
        # pixel Location
        p=np.array([idx[i,1]*5+4,idx[i,0]*5+4,1],dtype=float)
        # theta and phi (pixel gradient)
        Theta=vp2dir(K, R, p)
        phi=Gdir[idx[i,0],idx[i,1]]
        error=remove_polarity(phi-Theta)
        # P ang
        P_ang[i,:3]=np.nan_to_num(scipy.stats.norm(mu, sig).pdf(error))
        P_ang[i,3]=1/(2.0*np.pi)
    P_ang=P_ang*(np.array(P_m_prior))
    w_pm=P_ang
    Z_p=np.sum(w_pm,axis=1)
    Z_p=np.repeat(np.array((1./Z_p)), repeats=4).reshape((idx.shape[0],4))
    w_pm=w_pm*Z_p
    return w_pm

def M_step(S0, w_pm, idx, Gdir):
    '''
    :param S0 : the camera rotation parameters from the previous step
    :param w_pm : weights from E-step
    :return: R_m : the optimized camera rotation matrix
    '''

    def error_fun(S, w_pm):
        '''
        :param c : the variable we are going to optimize over
        :param w_pm : weights from E-step
        :return: error : the error we are going to minimize
        '''

        error = 0.0    # initial error setting to zero
        R = vector2matrix(S) # Note that the 'S' is just for optimization, it has to be converted to R during computation

        Weighted_Least_Sq=np.zeros(shape=(idx.shape[0],))
        for i in range(idx.shape[0]):
            # pixel Location
            p=np.array([idx[i,1]*5+4,idx[i,0]*5+4,1],dtype=float)
            # theta and phi (pixel gradient)
            Theta=vp2dir(K, R, p)
            phi=Gdir[idx[i,0],idx[i,1]]
            error=remove_polarity(phi-Theta)
            error=remove_polarity(error**2)
            w_pm_temp=w_pm[i,:3]
            # weighted Least Square
            Weighted_Least_Sq[i]=w_pm_temp.dot(error)
        error=np.sum(Weighted_Least_Sq)

        return error

    S_m = least_squares(error_fun, S0, args= (w_pm,))

    return S_m


