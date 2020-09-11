# MIT License
#
# Copyright (c) 2020 Sean G Polun (University of Missouri)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gp
import numba
from shapely.geometry import LineString
import json
import meshio
import warnings
import datetime
import trimesh
import csv
import data_model
nsims = 10000


class Plane(object):
    """
    Plane class, relays orientation and uncertainty. Initialization requires strike and dip, strike_unc and dip_unc are
    optional.

    Attributes
    ----------
    strike : float
        Strike azimuth
    dip: float
        dip angle
    strike_unc : float
        strike uncertainty
    dip_unc : float
        dip uncertainty
    pole : np.ndarray
        Pole to plane
    pole_unc : np.ndarray
        Uncertainties of pole components

    """

    def __init__(self, strike, dip, **kwargs):
        default_unc = 0.05
        n_sim = 25000
        self.strike = float(strike)
        self.dip = float(dip)
        if 'dip_unc' in kwargs.values():
            self.dip_unc = float(kwargs['dip_unc'])
        else:
            self.dip_unc = self.dip * default_unc
        if 'strike_unc' in kwargs.values():
            self.strike_unc = float(kwargs['strike_unc'])
        else:
            self.strike_unc = self.strike * default_unc

        rand_strike = (np.random.randn(n_sim) * self.strike_unc) + self.strike
        rand_dip = (np.random.randn(n_sim) * self.dip_unc) + self.dip
        rand_dip_dir = rand_strike + (math.pi / 2.)
        v1 = np.asarray([np.sin(rand_strike) * np.cos(rand_dip_dir), np.cos(rand_strike) * np.cos(rand_dip_dir),
                         np.sin(rand_dip_dir)])
        v2 = np.asarray([np.sin(rand_dip_dir) * np.cos(rand_dip), np.cos(rand_dip_dir) * np.cos(rand_dip),
                         np.sin(rand_dip)])
        fault_plane_pole_rand = np.cross(v1.T, v2.T)
        pole1 = np.mean(fault_plane_pole_rand, axis=0)
        self.pole = pole1 / np.linalg.norm(pole1)
        self.pole_unc = np.std(fault_plane_pole_rand, axis=0)


def tsurf_read(infile):
    """
    Will convert a GoCAD TSurf file (.ts extension) into a ply format. Note: Contains no sanity checks for input file
    besides file extension.
    Parameters
    ----------
    infile : str
        Path to input file

    Returns
    -------
    outfile: str
        Full path to output ply file
    """
    basename, ext = os.path.splitext(infile)
    if ext != '.ts':
        raise ImportError('File is not GoCAD Tsurf')
    in_file_open = open(infile, "r+")
    outfile = basename + ".ply"
    raw_text = in_file_open.readlines()
    vertices = []
    faces = []
    for line in raw_text:
        line_split = line.split()
        if line_split[0] == 'VRTX':
            vertices.append(line_split[2:5])
        elif line_split[0] == 'TRGL':
            faces.append(line_split[1:4])
    # out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # trimesh.exchange.export.export_mesh(out_mesh, outfile, file_type='ply')
    vertic_np = np.asarray(vertices, dtype=float)
    face_np = np.asarray(faces, dtype=int)
    cells = [("triangle", face_np)]
    meshio.write_points_cells(outfile, vertic_np, cells)
    return outfile


def lineament_from_mesh(in_meshes, depth, outfile, epsg):
    """

    Parameters
    ----------
    epsg
    in_meshes
    depth
    outfile

    Returns
    -------

    """
    lin_list = []
    name_list = []
    for mesh_num in range(len(in_meshes)):
        mesh_path = in_meshes[mesh_num]
        mesh_name = os.path.basename(mesh_path)
        out_lin_name = mesh_name + str(depth)
        mesh = trimesh.load_mesh(mesh_path)
        line = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, depth])
        line_dim = line.shape
        line_flat = line.reshape(line_dim[0]*line_dim[1], line_dim[2])
        line_flat_unq = np.unique(line_flat, axis=0)
        line_geom = LineString(line_flat_unq[:, 0:2])
        name_list.append(out_lin_name)
        lin_list.append(line_geom)
    geom_dict = {'Name': name_list, 'geometry': lin_list}
    out_gdf = gp.GeoDataFrame(geom_dict, crs=epsg)
    out_gdf.to_file(filename=outfile, driver='ESRI Shapefile')



@numba.njit()
def calc_normal_points(p0, p1, p2):
    """
    Calculates plane normal for the three points defining a triangular plane.

    Parameters
    ----------
    p0 : numpy.ndarray
        a 1x3 vector of coordinates (cartesian)
    p1 : numpy.ndarray
        a 1x3 vector of coordinates (cartesian)
    p2 : numpy.ndarray
        a 1x3 vector of coordinates (cartesian)

    Returns
    -------
    normal : numpy.ndarray
        a 1x3 unit vector
    midpoint : numpy.ndarray
        a 1x3 vector of coordinates

    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    u = [x1-x0, y1-y0, z1-z0]
    v = [x2-x0, y2-y0, z2-z0]
    cx = (x0 + x1 + x2)/3
    cy = (y0 + y1 + y2)/3
    cz = (z0 + z1 + z2)/3
    midpoint = np.array([cx, cy, cz])
    normal = np.cross(u, v)
    return normal, midpoint


@numba.njit(parallel=True)
def generate_mesh_normals(points, faces):
    """
    Generate normals and triangle midpoints for triangular mesh.
    Parameters
    ----------
    points : numpy.ndarray
        nx3 array with points in xyz cartesian coordinates (float)
    faces : numpy.ndarray
        nx3 array with indices for each point forming a triangular face (int)

    Returns
    -------
    normals : numpy.ndarray
        nx3 array of surface normals
    midpoints : numpy.ndarray
        nx3 array of triangular face midpoints

    """
    num_face = faces.shape[0]
    out_shape = faces.shape
    normals = np.empty(out_shape)
    midpoints = np.empty(out_shape)
    for i in numba.prange(num_face):
        face = faces[i]
        p0_ind, p1_ind, p2_ind = face
        p0 = points[p0_ind]
        p1 = points[p1_ind]
        p2 = points[p2_ind]
        normal, midpoint = calc_normal_points(p0, p1, p2)
        normals[i] = normal
        midpoints[i] = midpoint

    return normals, midpoints


@numba.njit
def rot_matrix_axis_angle(axis, angle):
    """
    Creates a 3x3 rotation matrix from rotation axis vector + rotation angle (in radians). Positive sign of input angle
    indicates a rotation in the counterclockwise direction.

    Parameters
    ----------
    axis : numpy.ndarray
        1x3 vector indicating rotation axis
    angle : float
        rotation angle (in radians)

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix
    """
    a1 = axis[0]
    a2 = axis[1]
    a3 = axis[2]
    #    rot = math.radians(angle)
    ident = np.identity(3)
    out = np.outer(axis, axis)
    a_x = np.array([[0, -a3, a2], [a3, 0, -a2], [-a2, a1, 0]])
    rot_matrix = (math.cos(angle) * ident) + (math.sin(angle) * a_x) + (1 - math.cos(angle) * out)
    return rot_matrix


@numba.njit
def define_principal_stresses(sv1, depth, shmin1, shmax1, hminaz, hmaxaz, sv_unc=10, shmin_unc=10, shmax_unc=10,
                              v_tilt_unc=10, h_az_unc=10, is_3d=False):
    """
    Generates cauchy stress tensor from principal stress directions, assumes vertical stress in one direction
    rotates sigma1 (maximum principal stress) direction to plane normal

    Parameters
    ----------
    sv : float
        vertical stress at depth
    depth : float
        Depth of analysis (sv = sv_grad * depth)
    shmin : float
        minimum horizontal stress at depth
    shmax : float
        maximum horizontal stress at depth
    hminaz : float
        minimum horizontal stress direction
    hmaxaz : float
        maximum horizontal stress direction
    sv_unc : float
        Vertical stress uncertainty at depth
    shmax_unc : float
        maximum horizontal stress uncertainty
    shmin_unc : float
        Minimum horizontal stress uncertainty
    v_tilt_unc : float
        tilt uncertainty for vertical stress
    h_az_unc : float
        Horizontal stress orientation uncertainty
    is_3d : bool
        Is this model 3d? returns gradients if so

    Returns
    -------
    princ_stress_tensor :  numpy.ndarray
        3x3 stress tensor aligned to principal stress orientations
    princ_stress_unc : numpy.ndarray
    axis : numpy.ndarray
        Sigma-1 stress unit axis (which is rotated to align with plane for normal / shear stress analysis)
    axis_unc : numpy.ndarray
        uncertainty for sigma-1 unit axis
    axis_out : numpy.ndarray

    """
    nsim = nsims
    h_az_unc = math.radians(h_az_unc)
    v_tilt_unc = math.radians(v_tilt_unc)
    if round(abs(hmaxaz - hminaz), 0) != 90.:
        raise ValueError('hmin and hmax are not orthogonal')
    # TODO: Truly fix azimuth issue with stress. Currently add 90 degrees to max stress direction.
    hmaxaz = math.radians(hmaxaz) + (math.pi/2)

    if is_3d:
        sv1 = sv1 / depth
        shmax1 = shmax1 / depth
        shmin1 = shmin1 / depth
    axis_out = np.zeros((nsim, 3))
    stress_out = np.zeros((nsim, 3))
    for i in numba.prange(nsim):
        sv = sv_unc * np.random.randn() + sv1
        shmax = shmax_unc * np.random.randn() + shmax1
        shmin = shmin_unc * np.random.randn() + shmin1

        if sv1 > shmax1 > shmin1:
            sigma1 = sv
            sigma2 = shmax
            sigma3 = shmin
            az_rad = h_az_unc * np.random.randn() + hmaxaz
            az_dip_rad = v_tilt_unc * np.random.randn() + math.pi / 2.
            rotated_axis = np.asarray([math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                                       math.sin(az_dip_rad)])

        else:
            sigma1 = shmax
            az_rad = h_az_unc * np.random.randn() + hmaxaz
            az_dip_rad = v_tilt_unc * np.random.randn() + 0.  # average dip is 0
            if shmax1 > sv1 > shmin1:
                sigma2 = sv
                sigma3 = shmin
                rotated_axis = np.asarray([math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                                           math.sin(az_dip_rad)])
            elif shmax1 > shmin1 > sv1:
                sigma2 = shmin
                sigma3 = sv
                rotated_axis = np.asarray([math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                                           math.sin(az_dip_rad)])
            else:
                raise ValueError('Unable to resolve principal stress orientations')
        axis_out[i, :] = rotated_axis
        stress_out[i, :] = np.asarray([sigma1, sigma2, sigma3])
    sigma1_mean = np.mean(stress_out[:, 0])
    sigma2_mean = np.mean(stress_out[:, 1])
    sigma3_mean = np.mean(stress_out[:, 2])
    sigma1_std = np.std(stress_out[:, 0])
    sigma2_std = np.std(stress_out[:, 1])
    sigma3_std = np.std(stress_out[:, 2])
    rotated_axis = np.array([np.mean(axis_out[:, 0]), np.mean(axis_out[:, 1]), np.mean(axis_out[:, 2])])
    rotated_axis = rotated_axis / np.linalg.norm(rotated_axis)
    axis_std = np.array([np.std(axis_out[:, 0]), np.std(axis_out[:, 1]), np.std(axis_out[:, 2])])
    princ_stress_tensor = np.array([[sigma1_mean, 0., 0.], [0., sigma2_mean, 0.], [0., 0., sigma3_mean]])
    princ_stress_tensor_unc = np.array([[sigma1_std, 0., 0.], [0., sigma2_std, 0.], [0., 0., sigma3_std]])
    # rotated_axis = np.array(rotated_axis)
    return princ_stress_tensor, rotated_axis, princ_stress_tensor_unc, axis_std, axis_out


@numba.njit
def rotate_plane_stress(sigma_1_ax, plane_norm, princ_stress_tensor):
    """
    Rotate principal stress tensor (3x3) onto specified plane.

    Parameters
    ----------
    sigma_1_ax : numpy.ndarray
        1x3 unit axis indicating the orientation of the maximum principal stress
    plane_norm : numpy.ndarray
        1x3 unit vector indicating normal of the plane to rotate stress onto
    princ_stress_tensor : numpy.ndarray
        3x3 matrix with 3 principal stresses on diagonal

    Returns
    -------
    numpy.ndarray
        3x3 matrix of resolved stress components on plane
    """
    rot_axis = np.cross(plane_norm, sigma_1_ax)
    rot_angle = -1 * np.dot(plane_norm, sigma_1_ax)
    rotmatrix = rot_matrix_axis_angle(rot_axis, rot_angle)
    plane_stress_tensor = rotmatrix.T @ princ_stress_tensor @ rotmatrix
    return plane_stress_tensor


@numba.njit
def point_az(x1, x2, y1, y2):
    """
    Determines azimuth between two points (in cartesian coordinates).

    Parameters
    ----------
    x1 : float
        point 1 x
    x2 : float
        point 2 x
    y1 : float
        point 1 y
    y2 : float
        point 2 y

    Returns
    -------
    float
        Azimuth between points (in degrees)

    """

    azimuth1 = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    azimuth = (azimuth1 + 360) % 360
    return azimuth


@numba.jit(forceobj=True)
def redistribute_vertices(geom, min_distance):
    """
    Redistribute vertices along a line into a specified spacing.

    Parameters
    ----------
    geom : shapely.geometry.LineString
        Shapely geometry linestring object
    distance : float
        Distance between line nodes

    Returns
    -------
    shapely.geometry.LineString
        Re-spaced linestring

    """
    node_multiplier = 101
    if geom.geom_type == 'LineString':
        init_nodes = geom.coords.__len__()
        mult_nodes = init_nodes * node_multiplier
        max_num_vert = int(round(geom.length / min_distance))
        if mult_nodes > max_num_vert:
            num_vert = max_num_vert
        else:
            num_vert = mult_nodes
        if num_vert == 0:
            num_vert = init_nodes
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, min_distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


@numba.njit
def deterministic_slip_tend(pole, stress_tensor, axis, pf, mu):
    """
    Compute slip tendency by deterministic methods (e.g. Morris et al., 1996, Geology)

    Parameters
    ----------

    pole : numpy.ndarray
        1x3 vector indicating pole to fault plane
    stress_tensor : numpy.ndarray
        3x3 matrix with principal stresses
    axis : numpy.ndarray
        1x3 vector indicating principal stress direction
    pf : float
        best guess pore fluid pressure
    mu : float
        best guess static friction coeffecient
    Returns
    -------
    stable_slip_tend : float
        slip tendency for input parameters + best guess pore fluid pressure
    pf1 : float
        Fluid pressure at which the failure criteria (Slip Tend > mu) is met

    """
    pf_step = 5.  # 5 MPa
    plane_stress = rotate_plane_stress(axis, pole, stress_tensor)
    sigma_n = plane_stress[0, 0]
    sigma_t = plane_stress[2, 0]
    slip_tendency = sigma_t / sigma_n
    sigma_n_eff = sigma_n - pf
    slip_tendency_eff = sigma_t / sigma_n_eff
    pf1 = pf
    while slip_tendency_eff < mu:
        pf1 = pf1 + pf_step
        sigma_n_eff = sigma_n - pf1
        slip_tendency_eff = sigma_t / sigma_n_eff
    return slip_tendency, pf1


@numba.njit(numba.float64[:, :](numba.float64[:]))
def ecdf(indata):
    """
    Generate an empirical cumulative density function (ECDF) for a set of data that has been filtered by a failure
    criterion.

    Parameters
    ----------
    indata : numpy.ndarray
        1xn array of values that meet failure criteria

    Returns
    -------
    out : numpy.ndarray
        2xn array of values and probabilities (val, prob)

    """
    # x = np.sort(data)
    if indata.size == 0:
        x = np.array([0., 0., 0.])
        y = np.array([0., 0.5, 1.])
        out = np.column_stack((x, y))
        return out
        # raise ValueError('Empty Dataset Given')
    # x = np.asarray(indata, dtype=np.float_)
    x = indata
    x.sort()
    n = x.size
    y = np.linspace(1.0 / nsims, 1, n)
    out = np.column_stack((x, y))
    return out


@numba.njit(parallel=True)
def monte_carlo_slip_tendency(pole, pole_unc, stress_tensor, stress_unc, axis, axis_unc, pf, mu,
                              mu_unc):
    """
    Computes fault slip tendency for a specific planar node defined by a pole.
    Parameters
    ----------
    pole : numpy.ndarray
        Pole to fault plane (1x3)
    pole_unc : numpy.ndarray
        Pole uncertainty (1x3)
    stress_tensor : numpy.ndarray
        3x3 array with principal stresses
    stress_unc : numpy.ndarray
        3x3 array with stress uncertainty
    axis : numpy.ndarray
        1x3 vector with sigma-1 orientation
    axis_unc : numpy.ndarray
        uncertainty for sigma-1 orientation
    pf : float
        Best Guess pore fluid pressure at depth
    mu : float
        Coeffecient of static friction for fault
    mu_unc : float
        Uncertainty for mu

    Returns
    -------
    out_data : numpy.ndarray
        nx3 array [fluid pressure, mu, slip tendency]
    """

    # n_sims = 10000
    pf_range = 25.
    # initialize uncertainty bounds
    princ_stress_vec = np.array([stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2]])
    princ_stress_unc = np.array([stress_unc[0, 0], stress_unc[1, 1], stress_unc[2, 2]])

    # main simulation loop
    out_data = np.empty((nsims, 3))
    for i in numba.prange(nsims):
        pole_rand = np.random.randn(3)
        pole1 = (pole_unc * pole_rand) + pole
        pole1 = pole1 / np.linalg.norm(pole1)

        stress_rand = np.random.randn(3)
        stress_rand = (princ_stress_unc * stress_rand) + princ_stress_vec

        stress1 = stress_rand * np.identity(3)

        axis_rand = np.random.randn(3)
        axis1 = (axis_unc * axis_rand) + axis
        axis1 = axis1 / np.linalg.norm(axis1)
        pf1 = np.random.random() * (pf + pf_range)
        mu1 = (np.random.randn() * mu_unc) + mu

        plane_stress = rotate_plane_stress(axis1, pole1, stress1)
        sigma_n = plane_stress[0, 0]
        sigma_t = plane_stress[2, 0]
        # slip_tendency = sigma_t / sigma_n
        sigma_n_eff = sigma_n - pf1
        slip_tendency_eff = sigma_t / sigma_n_eff
        out_data[i, 0] = pf1
        out_data[i, 1] = mu1
        out_data[i, 2] = slip_tendency_eff
    return out_data


# @numba.jit(forceobj=True, parallel=True)
def slip_tendency_2d(infile, input_model, input_params, dump_for_fsp=False):
    """
    Compute a  2d (i.e. where the 2d geometry is only known) slip tendency analysis. Outputs a map, as well
    as a table with the following schema:
    (ID : int) (x : float) (y : float) (Slip tendency : float) (Effective slip tendency : float)

    Parameters
    ----------
    dump_for_fsp
    infile : str
        Path to ESRI shapefile (or compatible with geopandas). Should be line objects with a 2d geometry.
    inparams : dict
        Dictionary containing fields: shmax, shmin, shmax, shmaxaz, shminaz, dip
    mode : str
        Mode flag for analysis type. det: deterministic analysis, mc: monte carlo / qra type analysis

    Returns
    -------
    numpy.ndarray

    """

    depth = input_params['depth']
    mode = input_params['mode']
    stress = input_params['stress']
    if mode == "mc":
        fail_threshold = input_params['fail_percent']
    else:
        fail_threshold = np.nan
    # min_node_distance = 150  # 50 m
    dip = input_model.Dip
    sv = input_model.Sv
    print(stress)

    if stress == "Reverse":
        shmax = input_model.ShMaxR
        shmin = input_model.ShMinR
    elif stress == "Normal":
        shmax = input_model.ShMaxN
        shmin = input_model.ShMinN
    elif stress == "Strike-Slip":
        shmax = input_model.ShMaxSS
        shmin = input_model.ShMinSS
    else:
        raise ValueError("Stress field is not defined.")

    shmaxaz = input_model.ShMaxAz
    shminaz = input_model.ShMinAz
    hydrostatic_pres = input_model.SHydro.mean * depth

    # shmax = inparams['shmax']
    # shmin = inparams['shmin']
    # sv = inparams['sv']
    # shmaxaz = inparams['shmaxaz']
    # shminaz = inparams['shminaz']
    # sv_unc1 = inparams['svunc']
    # az_unc = inparams['azunc']
    # shmin_unc = inparams['shmiunc']
    # shmax_unc = inparams['shMunc']

    # fail_threshold = inparams['fail_percent']
    # TODO: Implement this in GUI
    # hydrostatic_grad = inparams['hydrostatic_gradient']
    # hydrostatic_pres = hydrostatic_grad * depth
    # fail_threshold = fail_threshold / 100.

    if mode == 'mc':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std, _n = define_principal_stresses(sv.mean, depth, shmin.mean,
                                                                                    shmax.mean, shminaz.mean,
                                                                                    shmaxaz.mean, sv_unc=sv.std_unit(),
                                                                                    shmin_unc=shmin.std_unit(),
                                                                                    shmax_unc=shmax.std_unit(),
                                                                                    v_tilt_unc=10,
                                                                                    h_az_unc=shmaxaz.std_unit(),
                                                                                    is_3d=False)

    elif mode == 'det':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std, _n = define_principal_stresses(sv.mean, depth, shmin.mean,
                                                                                    shmax.mean, shminaz.mean,
                                                                                    shmaxaz.mean)

    else:
        raise ValueError('No recognized processing mode [''mc'' or ''det'']')
    lineaments = gp.GeoSeries.from_file(infile)
    bounds1 = lineaments.total_bounds
    #    lin_crs = lineaments.crs
    num_features = len(lineaments)
    out_features_list = []
    if dump_for_fsp:
        fsp_dump = []
        dump_dip = 90.
        csv_file = input_params['dump_file']
    for i in range(num_features):
        work_feat = lineaments[i]
        # work_feat_interp = redistribute_vertices(work_feat, min_node_distance)
        # work_feat_inter_coords = work_feat_interp.coords
        # num_nodes = len(work_feat_inter_coords)
        work_feat_coords = work_feat.coords
        num_nodes = len(work_feat_coords)
        num_segs = num_nodes - 1
        fault_out_arr = []
        for j in range(num_segs):
            metadata = {"line_id": i, "seg_id": j}
            # if j == 0:
            #     point1 = work_feat_inter_coords[j]
            #     point2 = work_feat_inter_coords[j + 1]
            # elif j == (num_nodes - 1):
            #     point1 = work_feat_inter_coords[j - 1]
            #     point2 = work_feat_inter_coords[j]
            # else:
            #     point1 = work_feat_inter_coords[j - 1]
            #     point2 = work_feat_inter_coords[j + 1]
            point1 = work_feat_coords[j]
            point2 = work_feat_coords[j + 1]
            azimuth = math.radians(point_az(point1[0], point2[0], point1[1], point2[1]))
            dip_rad = math.radians(dip.mean)
            if dump_for_fsp:
                point_diff = (point1[0] - point2[0], point1[1] - point2[1])
                seg_len = math.sqrt(point_diff[0]**2 + point_diff[1]**2) / 1000.
                strike = math.degrees(azimuth)
                x_dump = point1[0] / 1000
                y_dump = point1[1] / 1000
                dump_row = [x_dump, y_dump, strike, dump_dip, seg_len]
                fsp_dump.append(dump_row)
            if mode == 'det':

                fault_plane = Plane(azimuth, dip_rad, dip_unc=math.radians(dip.std_unit()))
                fault_plane_pole = fault_plane.pole
                slip_tend, fail_pressure = deterministic_slip_tend(fault_plane_pole, stress_tensor, sigma1_ax,
                                                                   input_model.max_pf, input_model.Mu.mean)
                result = data_model.SegmentDet2dResult(work_feat_coords[j][0], work_feat_coords[j][1], work_feat_coords[j + 1][0],
                                                       work_feat_coords[j + 1][1], slip_tend, metadata)
            elif mode == 'mc':
                dip_unc = dip.std_unit()
                fault_plane = Plane(azimuth, dip_rad, dip_unc=math.radians(dip_unc))
                fault_plane_pole = fault_plane.pole
                fault_plane_unc = fault_plane.pole_unc
                st_out = monte_carlo_slip_tendency(fault_plane_pole, fault_plane_unc, stress_tensor, stress_unc,
                                                   sigma1_ax, sig_1_std, input_model.max_pf, input_model.Mu.mean,
                                                   input_model.Mu.std_perc)
                pf_out = st_out[:, 0]
                #true_data = pf_out[st_out[:, 2] > st_out[:, 1]]
                #tend_out = ecdf(true_data)
                #tend_out[:, 0] = tend_out[:, 0] - hydrostatic_pres
                #ind_fail = (np.abs(tend_out[:, 1] - fail_threshold)).argmin()
                #outrow = [j, work_feat_coords[j][0], work_feat_coords[j][1], work_feat_coords[j + 1][0],
                #          work_feat_coords[j + 1][1], tend_out[ind_fail, 0]]
                result = data_model.SegmentMC2dResult(work_feat_coords[j][0], work_feat_coords[j][1],
                                           work_feat_coords[j + 1][0], work_feat_coords[j + 1][1],
                                           pf_out - hydrostatic_pres, metadata)
            else:
                raise ValueError('Cannot resolve calculation mode / not implemented yet')
            # flat_out_features.append()
            # fault_out_arr.append(outrow)
            out_features_list.append(result)
        print(["Finished object# ", str(i + 1)])
    out_features = data_model.Results2D(out_features_list, cutoff=fail_threshold)
    # flat_out_features = np.array(flat_out_features)
    # plotmin = np.min(flat_out_features[:, 5])
    # plotmax = np.max(flat_out_features[:, 5])
    if dump_for_fsp:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(fsp_dump)
    return out_features


def slip_tendency_3d(in_meshes, input_model, input_params):
    """
    Calculate slip tendency for 3D mesh (enumerated in list).

    Parameters
    ----------
    in_meshes : str list
        List of file paths containing triangular meshes (.ts, .ply, .msh) for analysis
    inparams : dict
        Input parameters for model
    mode : str
        'det' for deterministic model, 'mc' for probabilistic model

    Returns
    -------

    """

    ref_depth = input_params['depth']
    datum = input_params['datum']
    mode = input_params['mode']
    stress = input_params['stress']
    if mode == "mc":
        fail_threshold = input_params['fail_percent']
    else:
        fail_threshold = np.nan
    # min_node_distance = 150  # 50 m
    dip = input_model.Dip
    sv = input_model.Sv

    if stress == "reverse":
        shmax = input_model.ShMaxR
        shmin = input_model.ShMinR
    elif stress == "normal":
        shmax = input_model.ShMaxN
        shmin = input_model.ShMinN
    elif stress == "strike-slip":
        shmax = input_model.ShMaxSS
        shmin = input_model.ShMinSS
    else:
        raise ValueError("Stress field is not defined.")

    shmaxaz = input_model.ShMaxAz
    shminaz = input_model.ShMinAz
    hydrostatic_grad = input_model.SHydro.mean
    # shmax = inparams['shmax']
    # shmin = inparams['shmin']
    # sv = inparams['sv']
    # shmaxaz = inparams['shmaxaz']
    # shminaz = inparams['shminaz']
    # # TODO: Implement % uncertainty
    # sv_unc1 = inparams['svunc']
    # az_unc = inparams['azunc']
    # shmin_unc = inparams['shmiunc']
    # shmax_unc = inparams['shMunc']
    # datum = inparams['datum']
    # ref_depth = inparams['depth']
    # # TODO: Implement this in GUI
    # hydrostatic_grad = inparams['hydrostatic_gradient']

    # initialize attribute names for analysis
    # attr_50 = "PoreFluid_50"
    # attr_75 = "PoreFluid_75"
    # attr_99 = "PoreFluid_99"
    prob_levels = [
        15,
        18,
        20,
        22,
        25
    ]

    if mode == 'mc':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std = define_principal_stresses(sv.mean, ref_depth, shmin.mean,
                                                                                    shmax.mean, shminaz.mean,
                                                                                    shmaxaz.mean, sv_unc=sv.std_unit(),
                                                                                    shmin_unc=shmin.std_unit(),
                                                                                    shmax_unc=shmax.std_unit(),
                                                                                    v_tilt_unc=10,
                                                                                    h_az_unc=shmaxaz.std_unit(),
                                                                                    is_3d=True)

    elif mode == 'det':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std = define_principal_stresses(sv.mean, ref_depth, shmin.mean,
                                                                                    shmax.mean, shminaz.mean,
                                                                                    shmaxaz.mean)

    else:
        raise ValueError('No recognized processing mode [''mc'' or ''det'']')

    # convert stress tensors into gradients / km
    # TODO: Fix this shit
    stress_grad_tensor = stress_tensor / ref_depth
    # stress_grad_unc = stress_unc / ref_depth

    for mesh_num in range(len(in_meshes)):
        mesh_path = in_meshes[mesh_num]
        base_path, ext = os.path.splitext(mesh_path)
        # handle gocad tsurf files (UGH)
        if ext == '.ts':
            outmesh = tsurf_read(mesh_path)
            mesh = meshio.read(outmesh)
        elif (ext == '.ply') | (ext == '.msh'):
            mesh = meshio.read(mesh_path)
        else:
            warnings.warn('Not expected mesh format, use .ply, .msh, or .ts')
            try:
                mesh = meshio.read(mesh_path)
            except RuntimeError:
                warnings.warn(['Unable to load mesh: ' + mesh_path + '. Skipping.'])
                continue
            except OSError:
                warnings.warn('Unable to find mesh: ' + mesh_path + '. Skipping.')
                continue

        faces = mesh.cells_dict['triangle']
        points = mesh.points
        num_faces = faces.shape[0]
        normals, centroids = generate_mesh_normals(points, faces)
        # Extract mesh normals
        # mesh.add_attribute("face_normal")
        # normals = np.reshape(mesh.get_attribute("face_normal"), (num_faces, 3))
        # Extract face centroids
        # mesh.add_attribute("face_centroid")
        # centroids = np.reshape(mesh.get_attribute("face_centroid"), (num_faces, 3))

        # initialize output attributes
        # pf_50_prob = np.empty(num_faces, dtype=float)
        # pf_75_prob = np.empty(num_faces, dtype=float)
        # pf_99_prob = np.empty(num_faces, dtype=float)
        attrib_dict = {}
        attrib_names = []
        for prob in prob_levels:
            attr_name = "ProbLevel_" + str(prob)
            attrib_names.append(attr_name)
            attrib_dict[attr_name] = [np.empty(num_faces, dtype=float)]

        for face_num in range(num_faces):
            face = faces[face_num]
            p1 = points[face[0]]
            p2 = points[face[1]]
            p3 = points[face[2]]
            fault_plane_pole = normals[face_num]
            tri_center = centroids[face_num]
            # convert Z AMSL to Z depth (deeper is positive)
            depth = ((tri_center[2] * -1) + datum) / 1000

            stress_tensor = stress_grad_tensor * depth
            hydrostatic_pres = hydrostatic_grad * depth
            # stress_unc = stress_grad_unc * depth
            if mode == 'det':
                slip_tend, fail_pressure = deterministic_slip_tend(fault_plane_pole, stress_tensor, sigma1_ax,
                                                                   input_model.max_pf, input_model.Mu.mean)
                outrow = [mesh_num, face_num, slip_tend, fail_pressure]
            elif mode == 'mc':
                pole_unc = 0.05
                fault_plane_unc = fault_plane_pole * pole_unc
                st_out = monte_carlo_slip_tendency(fault_plane_pole, fault_plane_unc, stress_tensor, stress_unc,
                                                   sigma1_ax, sig_1_std, input_model.max_pf, input_model.Mu.mean,
                                                   input_model.Mu.std_perc)
                pf_out = st_out[:, 0]
                result = data_model.MeshFaceResult(face_num, face, p1, p2, p3, pf_out - hydrostatic_pres)
                # true_data = pf_out[st_out[:, 2] > st_out[:, 1]]
                # tend_out = ecdf(true_data)
                # tend_out[:, 0] = tend_out[:, 0] - hydrostatic_pres
                # ind_50 = (np.abs(tend_out[:, 1] - 0.5)).argmin()
                # ind_75 = (np.abs(tend_out[:, 1] - 0.75)).argmin()
                # ind_99 = (np.abs(tend_out[:, 1] - 0.99)).argmin()
                #
                # pf_50_prob[face_num] = tend_out[ind_50, 0]
                # pf_75_prob[face_num] = tend_out[ind_75, 0]
                # pf_99_prob[face_num] = tend_out[ind_99, 0]
                # loop to generate output attributes
                for prob in prob_levels:
                    # inds = (np.abs(tend_out[:, 1] - (prob / 100))).argmin()
                    attr_name = attrib_names[prob_levels.index(prob)]
                    attrib_dict[attr_name][0][face_num] = result.ecdf_cutoff(prob)

            else:
                raise ValueError('Cannot resolve calculation mode / not implemented yet')

        output_mesh = meshio.Mesh(points, [("triangle", faces)], cell_data=attrib_dict)
        # mesh.add_attribute(attr_50)
        # mesh.add_attribute(attr_75)
        # mesh.add_attribute(attr_99)
        #
        # mesh.set_attribute(attr_50, pf_50_prob)
        # mesh.set_attribute(attr_75, pf_75_prob)
        # mesh.set_attribute(attr_99, pf_99_prob)

        out_mesh_file = base_path + '_proc_norm' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.msh'
        # pymesh.save_mesh(out_mesh_file, mesh, *mesh.get_attribute_names())
        output_mesh.write(out_mesh_file)
    return


def plot_all(out_features, model_input, input_parameters):
    xmin = out_features.xmin
    xmax = out_features.xmax
    ymin = out_features.ymin
    ymax = out_features.ymax
    plotmin = out_features.plotmin
    plotmax = out_features.plotmax
    cutoff = out_features.cutoff
    flag = input_parameters['mode']
    # xmin = plot_bounds[0]
    # xmax = plot_bounds[2]
    # ymin = plot_bounds[1]
    # ymax = plot_bounds[3]
    # increase bounds by set percentage
    increase_scale = 0.25
    xrange = xmax - xmin
    yrange = ymax - ymin
    xinc = (xrange * increase_scale) / 2
    yinc = (yrange * increase_scale) / 2
    xmin1 = xmin - xinc
    xmax1 = xmax + xinc
    ymin1 = ymin - yinc
    ymax1 = ymax + yinc

    if flag == 'det':
        plotmin1 = plotmin - 0.01
        plotmax1 = plotmax + 0.01
        norm1 = mpl.colors.Normalize(vmin=plotmin1, vmax=plotmax1)
    elif flag == 'mc':
        plotmin1 = plotmin - 1
        plotmax1 = plotmax + 1
        # plotmin1 = 0.
        # plotmax1 = 5.
        norm1 = mpl.colors.Normalize(vmin=plotmin1, vmax=plotmax1)
    else:
        raise ValueError("Processing mode is not defined.")
    fig, ax = plt.subplots(dpi=300)
    # fig.set_size_inches(6, 4)
    ax.set_xlim(xmin1, xmax1)
    ax.set_ylim(ymin1, ymax1)
    for line in out_features.lines:
        # plot_data = np.array(out_features[i])
        # segs = np.unique([obj.seg_id for obj in line])
        n_seg = len(np.unique([obj.seg_id for obj in line]))
        segments = np.empty((n_seg, 2, 2))
        for j in range(n_seg):
            seg = line[j]
            p1 = np.asarray(seg.p1)
            p2 = np.asarray(seg.p2)
            points = np.vstack([p1, p2])
            # points = np.reshape(plot_data[j, 1:5], (2, 2))
            segments[j] = points
        # x1 = plot_data[:, 1].T
        # y1 = plot_data[:, 2].T
        # points = np.array([x1, y1]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if flag == 'det':
            slip_tendency = [obj.result for obj in line]
            # slip_tendency = plot_data[:, 5]
        elif flag == 'mc':
            # slip_tendency = plot_data[:, 5]
            slip_tendency = [obj.ecdf_cutoff(cutoff) for obj in line]
        else:
            raise ValueError("Processing mode is not defined.")

        lc = mpl.collections.LineCollection(segments, cmap=plt.get_cmap('jet_r'), norm=norm1)
        lc.set_array(np.array(slip_tendency))
        lc.set_linewidth(2)
        ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    if flag == 'det':
        axcb.set_label('Slip Tendency')
    elif flag == 'mc':
        axcb.set_label('Delta P over Hydrostatic to Failure [MPa]')
    ax2 = fig.add_axes([0.15, 0.1, 0.2, 0.2])
    str_img = plt.imread('./resources/h_stresses.png')
    stress_im = ax2.imshow(str_img)
    midx = str_img.shape[0] / 2
    midy = str_img.shape[1] / 2
    transf = mpl.transforms.Affine2D().rotate_deg_around(midx, midy, model_input.ShMaxAz.mean) + ax2.transData
    stress_im.set_transform(transf)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set(frame_on=False)
    plt.show()


def main(infile, params, model_params, dim='2d'):
    if dim == '2d':
        inputs = data_model.ModelInputs(params)
        results = slip_tendency_2d(infile, inputs, model_params, dump_for_fsp=False)
        plot_all(results, inputs, model_params)
    elif dim == '3d':
        inputs = data_model.ModelInputs(params)
        slip_tendency_3d(infile, params, model_params)
    else:
        raise(TypeError, "no defined type")



if __name__ == '__main__':
    test_params = {'depth': 2.5, 'mode': 'mc', 'stress': 'strike-slip', 'fail_percent': 0.5}
    in_file = 'test.json'
    with open(in_file) as json_file:
        j_data = json.load(json_file)
    inParams_test = j_data['input_data'][0]
    inFile_test = j_data['input_file']
    main(inFile_test, inParams_test, test_params)
