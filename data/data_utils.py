import numpy as np


def point_to_seg(x1, x2) -> np.ndarray:
    '''
    Method:
    -------
    Transform 2 points into a parametrized segment. Implicitely phi is in
    [-pi/2; pi/2], it is the oriented angle the segment makes with the
    horizontal line passing through its middle c.
    '''
    c = (x1[:2] + x2[:2])/2
    # TODO: funny could define different topologies to explore.
    if np.sum((x2-x1)**2) == 0:
        print('x2 is equal to x1?')
    r = np.sqrt(np.sum((x2-x1)**2))
    # TODO: chack that the angle is well oriented
    sign = np.sign(x2[0] - x1[0]) * np.sign(x2[1] - x1[1])
    phi = sign * np.arccos(np.abs(x2[0]-x1[0])/r)
    if phi < - np.pi/2 or phi > np.pi/2:
        raise ValueError('the value of phi is not in [-pi/2, pi/2] but it {}'.format(phi))
    res = np.hstack([c, r, phi])
    return res


def seg_to_point(seg) -> (np.ndarray, np.ndarray):
    '''transforms seg (c,r,phi) into a tuple of two 2-d points'''
    phi = seg[3]
    r = seg[2]
    c = seg[:2]
    dx = np.abs(np.cos(phi)*r/2)
    dy = np.abs(np.sin(phi)*r/2)
    x1 = c - np.array([dx, np.sign(phi)*dy])
    x2 = c + np.array([dx, np.sign(phi)*dy])
    return(x1, x2)
