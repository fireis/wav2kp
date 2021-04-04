from PIL import Image, ImageDraw
import numpy as np
import glob
# import dlib
import os
from scipy.optimize import curve_fit

# WARNING: this code is heavily based on vid2vid utils and processing functions
# Some functions were copied over, and the credit is to the authors of vid2vid
# https://github.com/NVIDIA/vid2vid

def read_img(img_path):
    return Image.open(img_path)


def read_keypoints(kp_path):
    return np.loadtxt(kp_path, delimiter=",")


def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_crop_coords(keypoints, size):
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    offset = (max_x - min_x) // 2
    min_y = max(0, min_y - offset * 2)
    min_x = max(0, min_x - offset)
    max_x = min(1920, max_x + offset)
    max_y = min(1080, max_y + offset)
    return (int(min_y), int(max_y), int(min_x), int(max_x))


def resize_img(img, keypoints):

    # hardcoded to comply with generated models, 256 and 320 should be variables
    new_w = 256
    new_h = 256

    min_y, max_y, min_x, max_x = get_crop_coords(keypoints, img.size)
    img = img.crop((min_x, min_y, max_x, max_y))
    new_w, new_h = make_power_2(new_w), make_power_2(new_h)
    method = Image.BICUBIC
    img = img.resize((256, 256), method)


    return img

def read_keypoints_v2v(A_path ):

    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                 [range(17, 22)],  # right eyebrow
                 [range(22, 27)],  # left eyebrow
                 [[28, 31], range(31, 36), [35, 28]],  # nose
                 [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                 [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                 [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                 [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                 ]
    label_list = [1, 2, 2, 3, 4, 4, 5, 6]

    keypoints = np.loadtxt(A_path, delimiter=',')

    # add upper half face by symmetry
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
    upper_pts = pts[1:-1, :].copy()
    upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

    return keypoints, part_list, label_list


def resize_orig_v2v(A_path):
    kp_path = A_path.replace('images', "keypoints").replace('png', 'txt')
    # print(f"kp_path: {kp_path}")
    reduced_img_path = A_path.replace("images", "images_reduced")

    keypoints, part_list, label_list = read_keypoints_v2v(kp_path)
    
    im_edges = Image.open(A_path)
    im_edges = resize_img(im_edges, keypoints)
    im_edges.save(reduced_img_path)
    # Image.fromarray(crop(im_edges, keypoints)).save(A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
    return

def linear(x, a, b):
    return a * x + b

def crop(img, keypoints):
    size = 256,256
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    xc = (min_x + max_x) // 2
    yc = (min_y * 3 + max_y) // 4
    h = w = (max_x - min_x) * 2.5
    xc = min(max(0, xc - w // 2) + w, size[0]) - w // 2
    yc = min(max(0, yc - h // 2) + h, size[1]) - h // 2
    min_x, max_x = int(xc - w // 2), int(xc + w // 2)
    min_y, max_y = int(yc - h // 2), int(yc + h // 2)

    if isinstance(img, np.ndarray):
        return img[min_y:max_y, min_x:max_x]
    else:
        return img.crop((min_x, min_y, max_x, max_y))

def func(x, a, b, c):
    return a * x**2 + b * x + c

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:

        if len(x) < 3:
            popt, _ = curve_fit(linear, x, y)
        else:
            popt, _ = curve_fit(func, x, y)
            if abs(popt[0]) > 1:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def draw_face_edges_v2v(A_path):
    keypoints, part_list, label_list = read_keypoints_v2v(A_path)
    w, h = 1920, 1080
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    # im_edges = np.zeros((h, w), np.uint8) # edge map for all edges


    color = 0
    color_edge = (255, 255, 255)

    im_edges = np.full((h, w), color, np.uint8)

    for edge_list in part_list:
        for edge in edge_list:

            # im_edge = np.full((h, w), color, np.uint8)
            im_edge = np.zeros((h, w), np.uint8)  # edge map for the current edge
            for i in range(0, max(1, len(edge) - 1),
                           edge_len - 1):  # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i + edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                curve_x, curve_y = interpPoints(x, y)  # interp keypoints to get the curve shape

                drawEdge(im_edges, curve_x, curve_y, color=color_edge)

    # np.save(im_edges,A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
    im_edges = Image.fromarray(im_edges)
    im_edges = resize_img(im_edges, keypoints)
    im_edges.save(A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
    # Image.fromarray(crop(im_edges, keypoints)).save(A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
    return