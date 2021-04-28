import pyrealsense2 as rs
import json
from express_id.pipe import setup_pipe
import open3d as o3d
import numpy as np
import os
import time
import copy
from scipy.spatial import KDTree

def main():

    with open('config.json') as f:
        config = json.load(f)

    # declare filters
    dec_filter = rs.decimation_filter()
    spat_filter = rs.decimation_filter()
    temp_filter = rs.temporal_filter()
    # setup pipeline
    pipeline, profile = setup_pipe()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)
    pc = rs.pointcloud()
    align_to = rs.stream.depth
    align = rs.align(align_to)
    depth_scale = depth_sensor.get_depth_scale()
    _runtime = 1
    cloud = []
    mode = 'start'
    try:
        while True:
            if mode == 'start':
                print('Command List: \n'
                      '"capture": this mode will allow you to collect data on 3D data of new holsters\n'
                      '"search": this mode will find a list of the most similar holsters\n'
                      '"exit": type exit to terminate the program')
                input_command = input("Enter command: ")
                if input_command == 'exit':
                    print('closing')
                    break
                elif input_command == 'capture':
                    print('capture: type the name of the holster and hit enter or else type "search" to switch to search mode')
                    mode = 'capture'
                elif input_command == 'search':
                    mode = 'search'
            elif mode == 'capture':
                input_command = input('enter holster to capture:')
                if input_command == 'exit':
                    print('closing')
                    break
                elif input_command == 'search':
                    print('search: place target holster in the target area and hit enter to begin or type "capture" to change to capture mode')
                    mode = 'search'
                else:
                    print(f'capturing holster data: in a moment a pointcloud will appear, file will not be saved until the pointcloud window is closed')
                    capture(input_command, align, pipeline, pc, dec_filter, spat_filter, temp_filter, True)
                    print(f'[ {input_command}.ply ] : file saved')

            elif mode == 'search':
                input_command = input('Press [enter] to begin search: ')
                if input_command == 'exit':
                    print('closing')
                    break
                elif input_command == '':
                    print('begin search')
                    directory = "express_id/store"
                    fit = []
                    best_fit = {'filename': str, 'SumOfSquares': 1000, 'result': {}}
                    for file in os.listdir(directory):
                        filename = os.fsdecode(file)
                        if filename != 'target.ply':
                            print(filename)
                            full_path = os.path.join(directory, filename)
                            source = o3d.io.read_point_cloud(full_path)
                        else:
                            continue
                        capture('target', align, pipeline, pc, dec_filter, spat_filter, temp_filter, False)
                        target = o3d.io.read_point_cloud("express_id/store/target.ply")
                        tic = time.perf_counter()

                        voxel_size = 0.001
                        source = source.voxel_down_sample(voxel_size)
                        target = target.voxel_down_sample(voxel_size)
                        flip_source = copy.deepcopy(source)
                        R = np.asarray([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]])
                        flip_source = flip_source.rotate(R, [0, 0, 0])

                        threshold = 10
                        trans_init = np.asarray([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])

                        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
                        reg_p2p2 = o3d.pipelines.registration.registration_icp(flip_source, target, threshold, trans_init,
                                                                               o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                               o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
                        result = copy.deepcopy(source)
                        result.transform(reg_p2p.transformation)
                        result_flip = copy.deepcopy(flip_source)
                        result_flip.transform(reg_p2p2.transformation)
                        r_count = len(result.points)
                        t_count = len(target.points)
                        if r_count > t_count:
                            tree = KDTree(target.points)
                            dist, _ = tree.query(result.points)
                            dist2, _ = tree.query(result_flip.points)
                        else:
                            tree = KDTree(result.points)
                            tree2 = KDTree(result_flip.points)
                            dist, _ = tree.query(target.points)
                            dist2, _ = tree2.query(target.points)
                        sum_of_squares = np.sum(dist**2)
                        sum_of_squares_flip = np.sum(dist2**2)
                        if sum_of_squares > sum_of_squares_flip:
                            sum_of_squares = sum_of_squares_flip
                            result = result_flip
                        print(f'registration sum of squares: '+str(sum_of_squares))
                        fit.append({"Type": filename, "SumOfSquares Error": sum_of_squares})
                        if best_fit['SumOfSquares'] > sum_of_squares:
                            best_fit['filename'] = filename
                            best_fit['SumOfSquares'] = sum_of_squares
                            best_fit['result'] = result

                    toc = time.perf_counter()
                    new_fit = sorted(fit, key=lambda i: i['SumOfSquares Error'], reverse=True)
                    print(f"time: " + str(toc-tic))
                    for ii in new_fit:
                        print(ii)
                    print('Best Fit: ' + best_fit['filename'])

                elif input_command == 'capture':
                    print('capture: type the name of the holster and hit enter\n'
                          'or else type "search" to switch to search mode')
                    mode = 'capture'
                else:
                    print('command not valid. Please use the "capture"/"exit" commands or press [enter] to begin search')
    finally:
        pipeline.stop()


def capture(name, align, pipeline, pc, dec_filter, spat_filter, temp_filter, open_image):
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    filtered = depth_frame
    filtered = dec_filter.process(filtered)
    filtered = spat_filter.process(filtered)
    filtered = temp_filter.process(filtered)

    color_frame = aligned_frames.get_color_frame()
    # Tell pointcloud object to map to this color frame
    pc.map_to(color_frame)

    # Generate the pointcloud and texture mappings
    points = pc.calculate(filtered)
    points.export_to_ply("temp.ply", color_frame)
    pcd = o3d.io.read_point_cloud("temp.ply")
    [xmin, ymin, zmin] = pcd.get_min_bound()
    [xmax, ymax, zmax] = pcd.get_max_bound()
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-.1, -.125, zmin),
                                               max_bound=(0.1, 0.125, zmax))

    pcd_crop = pcd.crop(bbox)

    [xmin, ymin, zmin] = pcd_crop.get_min_bound()
    [xmax, ymax, zmax] = pcd_crop.get_max_bound()
    #todo  comment the variable in the zmin box determines how high the flat is
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(xmin, ymin, -.31),
                                               max_bound=(xmax, ymax, zmax))
    pcd_crop2 = pcd_crop.crop(bbox)

    cloud = np.asarray(pcd_crop2.points)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(cloud)
    if open_image:
        o3d.visualization.draw(pointcloud)

    cwd = os.getcwd()
    filename = cwd+'/express_id/store/'+name+'.ply'
    o3d.io.write_point_cloud(filename, pointcloud)

if __name__=='__main__':
    main()