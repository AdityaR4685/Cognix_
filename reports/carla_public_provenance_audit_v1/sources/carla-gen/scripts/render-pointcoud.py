"""
Create renderings of 3d lidar pointsclouds.

For some reason, we can not change the viewpoint, so we have to rotate the pointcloud
"""

import open3d as o3d
import numpy as np
import copy
import click
import os
from os.path import join
import pandas as pd

from open3d.cuda.pybind.utility import Vector3dVector
from tqdm import tqdm

rgb_map = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
]


def read_ply(ply_file: str) -> pd.DataFrame:
    """
    Reads a ply file and returns a pandas dataframe
    :param ply_file:
    :return:
    """
    data = []
    header = []

    header_done = False

    with open(ply_file, "r") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if n in (0, 2):
                continue

            if not header_done and line == "end_header":
                header_done = True
                continue

            if not header_done:
                prop, dtype, name = line.split(" ")
                if prop != "property":
                    continue

                header.append(name)
                continue

            d = {k: float(v) for k, v in zip(header, line.split())}
            data.append(d)

    return pd.DataFrame(data)


def deg2rad(deg):
    return deg * np.pi / 180


def render_pointcloud_to_image(bin_file, bin_label_file, image_file, vis):
    labels = np.fromfile(bin_label_file, dtype=np.uint32).reshape(-1, 2)
    points = np.fromfile(bin_file, dtype=np.dtype("f4")).reshape(-1, 4)

    pcd = o3d.geometry.PointCloud()
    default_color = (0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
    colors = np.apply_along_axis(
        lambda x: rgb_map.get(x[1] - 1, default_color), axis=1, arr=labels
    )

    pcd.colors = Vector3dVector(colors)
    pcd.points = Vector3dVector(points[:, :3])

    # Rotating the point cloud about the X-axis by 90 degrees.
    x_theta = deg2rad(0)
    y_theta = deg2rad(0)
    z_theta = deg2rad(90)
    rotated_pcd = copy.deepcopy(pcd)
    rotation_matrix = rotated_pcd.get_rotation_matrix_from_axis_angle(
        [x_theta, y_theta, z_theta]
    )
    rotated_pcd.rotate(rotation_matrix, center=(0, 0, 0))

    points = Vector3dVector(np.array([[25, 25, 25], [-25, -25, -25]]))
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
    rotated_pcd = rotated_pcd.crop(bbox)
    vis.add_geometry(rotated_pcd)

    # car bbox
    points = Vector3dVector(np.array([[1, 1.5, 2], [-1, -2.5, 0]]))
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
    vis.add_geometry(bbox)

    view_control = vis.get_view_control()
    # view_control.set_zoom(0.2)  # Adjust zoom level; lower values are closer
    # view_control.set_lookat([0.5, 0.5, 0])
    # Set the camera height
    lookat_point = [0, 0, 0]  # Center point of the view
    camera_position = [0, -5, 5]  # Closer position
    up_vector = [0, 0, 1]  # Usually z-axis as up direction

    # Adjust camera settings
    view_control.set_lookat(lookat_point)
    view_control.set_front(np.array(camera_position) - np.array(lookat_point))
    view_control.set_up(up_vector)

    # image = vis.capture_screen_float_buffer(False)
    vis.capture_screen_image(image_file, do_render=True)

    vis.remove_geometry(rotated_pcd)
    vis.remove_geometry(bbox)


@click.command()
@click.argument("root")
@click.option("--output-dir", "-o", default="pcl-renders")
def main(root, output_dir="image"):
    """
    Render some pointsclouds in a directory
    """
    files = os.listdir(root)
    files = [f for f in files if f.endswith(".bin") and not f.startswith("labels")]
    files.sort()

    output_dir = join(root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Create a visualization
    vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
    # vis.create_window(width=1920, height=1080, visible=False)
    # vis.create_window(width=1024, height=1024, visible=False)
    vis.create_window(width=2048, height=2048, visible=False)

    # Set the background color
    opt = vis.get_render_option()
    # opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    opt.background_color = np.asarray([1, 1, 1])  # Dark gray background
    opt.point_size = 1.5  # Increase point size for better visibility

    for file in tqdm(files):
        out = join(output_dir, file.replace(".bin", ".jpg"))
        infile = join(root, file)

        # Example usage
        render_pointcloud_to_image(infile, join(root, f"labels-{file}"), out, vis)

    vis.destroy_window()


if __name__ == "__main__":
    main()
