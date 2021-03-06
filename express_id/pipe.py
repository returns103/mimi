import pyrealsense2 as rs
import numpy as np
import cv2

def setup_pipe():
    pipeline = rs.pipeline()
    _config = rs.config()

    _config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    _config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(_config)

    return pipeline, profile


def aligned(config, frames):
    color_frame = frames.get_color_frame()
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)
    aligned_depth_frame = frameset.get_depth_frame()

    colorizer = rs.colorizer()
    push_depth = np.asanyarray(aligned_depth_frame.get_data())

    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    color = np.asanyarray(color_frame.get_data())
    alpha = config['alpha']
    beta = 1 - alpha
    images = cv2.addWeighted(color, alpha, colorized_depth, beta, 0.0)

    low_threshold = 50
    ratio = 2  # recommend ratio 2:1 - 3:1
    kernal = 3
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_to_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    edges = cv2.Canny(depth_to_jet, low_threshold, low_threshold * ratio, kernal)
    # images = depth_image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    images = cv2.addWeighted(images, alpha, edges, beta, 0.0)

    return images, push_depth, color


def background(config, depth, mult):
    depth_data = depth.copy()

    # depth_data = np.asanyarray(depth.get_data())
    hist, bins = np.histogram(depth_data[depth_data != 0], bins=100)
    high_bin = bins[np.argmax(hist)]
    bin_width = (np.amax(bins) - np.amin(bins)) / 100
    dd = depth_data

    low = high_bin-bin_width*(mult-1)
    high = high_bin+bin_width*mult
    dd[dd > high] = 0
    dd[dd < low] = 0

    tbl_far = np.amax(dd[dd != 0])

    binary = dd.copy()
    binary[binary != 0] = 1
    # binary = cv2.applyColorMap(cv2.convertScaleAbs(binary, alpha=0.03), cv2.COLORMAP_JET)
    return dd, binary, tbl_far

def xyz(depth):
    pc = rs.pointcloud()
    points = pc.calculate(depth)
    pts = np.asanyarray(points.get_vertices())
    ap = [*zip(*pts)]
    x = ap[0]
    y = ap[1]
    z = ap[2]

    return x, y, z

def centroid(binary):

    flat = binary.sum(axis=0)
    peak_thresh=300
    min_width = 100
    counter = 0
    peaks =[]
    widths=[]
    for i, value in enumerate(flat):
        if value < peak_thresh:
            if counter >min_width:
                peaks.append(i-counter/2)
                widths.append(counter)
            counter = 0
        else:
            counter = counter+1

    ind = widths.index(max(widths))
    return peaks[ind], widths[ind]


def crop_horiz(image, center, width):
    l_bound = int(center-width/2)
    u_bound = int(center+width/2)
    out = image[:, l_bound:u_bound]
    return out, l_bound, u_bound

# cv2.polylines