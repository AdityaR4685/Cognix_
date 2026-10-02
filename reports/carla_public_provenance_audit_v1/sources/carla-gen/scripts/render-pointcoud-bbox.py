import json
from pathlib import Path

import click
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.utility import Vector3dVector
from open3d.geometry import get_rotation_matrix_from_xyz
from tqdm import tqdm
import pandas as pd

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

rgb_map = {n: v for n, v in enumerate(rgb_map)}


def bbox_camera2lidar(bboxes, tr_velo_to_cam, r0_rect):
    """
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    """
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    extended_xyz = np.pad(
        bboxes[:, :3], ((0, 0), (0, 1)), "constant", constant_values=1.0
    )
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    bboxes_lidar = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return np.array(bboxes_lidar, dtype=np.float32)


def camera2lidar_points(pts, tr_velo_to_cam, r0_rect):
    """
    Transform a set of 3D points from camera coordinate system to LiDAR coordinate system.

    Args:
        pts (np.ndarray): Shape (N, 3), points in camera coordinates.
        tr_velo_to_cam (np.ndarray): Transformation matrix (4x4) from LiDAR to camera.
        r0_rect (np.ndarray): Rectification matrix (4x4).

    Returns:
        np.ndarray: Transformed points in LiDAR coordinates, shape (N, 3).
    """
    # Extend points to homogeneous coordinates (N,4)
    extended_pts = np.pad(pts, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    pts_lidar = extended_pts @ rt_mat.T
    return pts_lidar[:, :3]


def read_calib(file_path, extend_matrix=True):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(" ")[1:]], dtype=np.float64).reshape(
        3, 4
    )
    P1 = np.array([item for item in lines[1].split(" ")[1:]], dtype=np.float64).reshape(
        3, 4
    )
    P2 = np.array([item for item in lines[2].split(" ")[1:]], dtype=np.float64).reshape(
        3, 4
    )
    P3 = np.array([item for item in lines[3].split(" ")[1:]], dtype=np.float64).reshape(
        3, 4
    )

    R0_rect = np.array(
        [item for item in lines[4].split(" ")[1:]], dtype=np.float64
    ).reshape(3, 3)
    Tr_velo_to_cam = np.array(
        [item for item in lines[5].split(" ")[1:]], dtype=np.float64
    ).reshape(3, 4)
    Tr_imu_to_velo = np.array(
        [item for item in lines[6].split(" ")[1:]], dtype=np.float64
    ).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate(
            [Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0
        )
        Tr_imu_to_velo = np.concatenate(
            [Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0
        )

    calib_dict = dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo,
    )
    return calib_dict


def deg2rad(deg):
    return deg * np.pi / 180


def set_for_keys(my_dict, key_arr, val):
    """
    Set val at path in my_dict defined by the string (or serializable object) array key_arr
    """
    current = my_dict
    for i in range(len(key_arr)):
        key = key_arr[i]
        if key not in current:
            if i==len(key_arr)-1:
                current[key] = val
            else:
                current[key] = {}
        else:
            if type(current[key]) is not dict:
                print("Given dictionary is not compatible with key structure requested")
                raise ValueError("Dictionary key already occupied")

        current = current[key]

    return my_dict

def to_formatted_json(df, sep="."):
    result = []
    for _, row in df.iterrows():
        parsed_row = {}
        for idx, val in row.items():
            keys = idx.split(sep)
            parsed_row = set_for_keys(parsed_row, keys, val)

        result.append(parsed_row)
    return result



# ---------------------------------------------------------------------------
# helper: read all KITTI boxes from one *_extended.json
# ---------------------------------------------------------------------------
def load_kitti_bboxes(feather_path: Path, callib_file: Path, side="front", max_dist=50):
    """
    Return a list of open3d.geometry.OrientedBoundingBox objects that represent
    the KITTI 'kitti' entries in a *_extended.json file.

    Coordinate convention
    ---------------------
    The KITTI 'camera' frame (x→, y↓, z←) coincides with the lidar frame
    used in the Carla-derived dataset you render.  Therefore no additional
    axis permutation is carried out; only the user-defined global rotation
    later in the pipeline is applied.

    Parameters
    ----------
    json_file : pathlib.Path
        Path to the *_extended.json for this frame.

    Returns
    -------
    list[open3d.geometry.OrientedBoundingBox]
    """
    df = pd.read_feather(feather_path)
    objects = to_formatted_json(df, sep=".")

    # with open(json_file, "r") as fh:
    #    objects = json.load(fh)

    calib_info = read_calib(callib_file)
    tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
    r0_rect = calib_info["R0_rect"].astype(np.float32)
    # P2 = calib_info['P2'].astype(np.float32)

    obbs = []
    for obj in objects:
        kitti = obj["kitti"]

        # Create bboxes in camera coords: Nx7 => (loc_x, loc_y, loc_z, dim_x, dim_y, dim_z, rot_y)
        location = np.array(list(map(float, kitti["location"].split())))
        dimensions = np.array(list(map(float, kitti["dimensions"].split())))
        rotation_y = np.array(kitti["rotation_y"])

        rotation_y = np.expand_dims(rotation_y, 0)
        dimensions = np.expand_dims(dimensions, 0)
        location = np.expand_dims(location, 0)

        bboxes_camera = np.concatenate(
            [location, dimensions, rotation_y[:, None]], axis=-1
        )

        filtered_locs = bboxes_camera[:, 0:3]  # Nx3

        if np.sqrt((filtered_locs**2).sum()) > max_dist:
            continue

        # Also, convert the filtered locations (centers) to LiDAR coordinates.
        filtered_locs_lidar = camera2lidar_points(
            filtered_locs, tr_velo_to_cam, r0_rect
        )
        if side == "front":
            pass

        elif side == "rear":
            filtered_locs_lidar[0, 0] = -filtered_locs_lidar[0, 0]
            filtered_locs_lidar[0, 1] = -filtered_locs_lidar[0, 1]

        elif side == "right":
            # switch x and y and invert y (or x, whatever)
            tmp = filtered_locs_lidar[0, 1]
            # invert y axis
            filtered_locs_lidar[0, 1] = -filtered_locs_lidar[0, 0]
            filtered_locs_lidar[0, 0] = tmp

        elif side == "left":
            # switch x and y and invert y (or x, whatever)
            tmp = filtered_locs_lidar[0, 1]
            # invert y axis
            filtered_locs_lidar[0, 1] = filtered_locs_lidar[0, 0]
            filtered_locs_lidar[0, 0] = -tmp

            # filtered_locs_lidar[0, 0] = -filtered_locs_lidar[0, 0]
        else:
            raise ValueError()

        # x, y, z
        extent = [dimensions[0, 1], dimensions[0, 2], dimensions[0, 0]]

        R = get_rotation_matrix_from_xyz((0, 0, -rotation_y[0]))
        obb = o3d.geometry.OrientedBoundingBox(
            center=filtered_locs_lidar[0], R=R, extent=extent
        )

        # colour code by object class (here: pedestrian/car/other)
        cls = kitti["type"].lower()
        if "car" in cls:
            obb.color = (1.0, 0.0, 0.0)  # red
        elif "pedestrian" in cls:
            obb.color = (0.0, 0.8, 0.0)  # green
        elif "motorcycle" in cls:
            obb.color = (1.0, 1.0, 0.0)
        elif "citystatic" in cls or "citydynamic" in cls:
            # "cityroads" in cls
            obb.color = (0.0, 1.0, 1.0)
        elif "citytrafficsign" in cls or "citytrafficlight" in cls:
            obb.color = (0.5, 0.5, 0.5)
            # citydynamic
        else:
            continue

        obbs.append(obb)
    return obbs


def render_pointcloud_to_image(bin_file, bin_label_file, image_file, vis, bbox_list):
    # --- load raw data -----------------------------------------------------
    # labels  = np.fromfile(bin_label_file, dtype=np.uint32).reshape(-1, 2)
    labels = pd.read_feather(bin_label_file).values
    # points  = np.fromfile(bin_file,       dtype=np.float32).reshape(-1, 4)
    points = pd.read_feather(bin_file).values

    default_color = (0.619607843, 0.85490196, 0.89803922)
    colors = np.apply_along_axis(
        lambda x: rgb_map.get(x[1] - 1, default_color), axis=1, arr=labels
    )

    # --- populate point cloud ---------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points[:, :3])
    pcd.colors = Vector3dVector(colors)

    # --- common rotation for cloud, boxes, and axis ------------------------
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [deg2rad(0), deg2rad(0), deg2rad(90)]
    )
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    for obb in bbox_list:
        obb.rotate(rotation_matrix, center=(0, 0, 0))

    # ----- NEW: tri-arrow coordinate frame ---------------------------------
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    axis.rotate(rotation_matrix, center=(0, 0, 0))
    vis.add_geometry(axis)
    # -----------------------------------------------------------------------

    # --- crop, add geometry, and render -----------------------------------
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        Vector3dVector(np.array([[25, 25, 25], [-25, -25, -25]]))
    )
    pcd = pcd.crop(aabb)

    vis.add_geometry(pcd)
    for obb in bbox_list:
        vis.add_geometry(obb)

    vc = vis.get_view_control()
    vc.set_lookat([0, 0, 0])
    vc.set_front(np.array([0, -5, 5]))
    vc.set_up([0, 0, 1])

    vis.capture_screen_image(str(image_file), do_render=True)

    # --- clean-up ----------------------------------------------------------
    vis.remove_geometry(axis)  # remove axis first
    vis.remove_geometry(pcd)
    for obb in bbox_list:
        vis.remove_geometry(obb)


# ---------------------------------------------------------------------------
# revised: main()
# ---------------------------------------------------------------------------
@click.command()
@click.argument("root")  # directory with .bin clouds
# @click.argument("json_root")                  # directory with *_extended.json
@click.option("--output-dir", "-o", default="pcl-renders")
def main(root, output_dir, max_dist=50):
    """
    Render lidar frames and draw KITTI 3-D bounding boxes from *_extended.json.
    """
    root = Path(root)
    pcl_root = root / "pointclouds"

    # json_root = Path(json_root)

    bin_files = sorted(
        p for p in pcl_root.glob("*.feather") if not p.name.startswith("labels")
    )
    output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2048, height=2048, visible=False)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 1.5

    for bin_path in tqdm(bin_files):
        bbox_list = []
        # stem = "000058"
        stem = bin_path.stem

        for sensor in ["right", "left", "front", "rear"]: #
            feather_path = root / f"kitti-{sensor}" / "extended"/ f"{stem}.feather"
            calib_path = root / f"kitti-{sensor}" / "calib" / f"{stem}.txt"

            bb = load_kitti_bboxes(feather_path, calib_path, side=sensor)
            print(f"{sensor}[{stem}] -> {len(bb)}")
            bbox_list += bb

        lbl_path = pcl_root / f"labels-{stem}.feather"
        out_path = output_dir / f"{stem}.jpg"

        render_pointcloud_to_image(bin_path, lbl_path, out_path, vis, bbox_list)
    #  break
    vis.destroy_window()


if __name__ == "__main__":
    main()
