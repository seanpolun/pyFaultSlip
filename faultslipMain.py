#!/usr/bin/env python3

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gp
import numba
from shapely.geometry import LineString

nsims = 10000

# TODO: Implement 3D slip tendency analysis


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
        n_sim = 1500
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

    """
    nsim = 5000
    h_az_unc = math.radians(h_az_unc)
    v_tilt_unc = math.radians(v_tilt_unc)
    hmaxaz = math.radians(hmaxaz)
    hminaz = math.radians(hminaz)

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

        if abs(hmaxaz - hminaz) != 90:
            raise ValueError('hmin and hmax are not orthogonal')

        if sv > shmax > shmin:
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
            if shmax > sv > shmin:
                sigma2 = sv
                sigma3 = shmin
                rotated_axis = np.asarray([math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                                           math.sin(az_dip_rad)])
            elif shmax > shmin > sv:
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
    rotated_axis = np.mean(axis_out, axis=0)
    axis_std = np.std(axis_out, axis=0)
    princ_stress_tensor = np.array([[sigma1_mean, 0., 0.], [0., sigma2_mean, 0.], [0., 0., sigma3_mean]])
    princ_stress_tensor_unc = np.array([[sigma1_std, 0., 0.], [0., sigma2_std, 0.], [0., 0., sigma3_std]])
    # rotated_axis = np.array(rotated_axis)
    return princ_stress_tensor, rotated_axis, princ_stress_tensor_unc, axis_std

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
def monte_carlo_slip_tendency(pole, stress_tensor, stress_unc, axis, axis_unc, pf, mu, fault_plane_unc,
                              mu_unc):
    """

    Parameters
    ----------
    plane : Plane object
        Pole to fault plane (1x3)
    stress_tensor : numpy.ndarray
        3x3 array with principal stresses
    axis : numpy.ndarray
        1x3 vector with sigma-1 orientation
    pf : float
        Best Guess pore fluid pressure at depth
    mu : float
        Coeffecient of static friction for fault
    unc_bounds : float
        Uncertainty for bounds, presently 5% as default

    Returns
    -------

    """
    # TODO: Rewrite monte_carlo_slip_tendency() to take advantage of numba's CUDA support, may speed up simulation more,
    #  currently takes ~2s per full lineament simulation, may get bogged down on detailed 3d faults

    # n_sims = 10000
    pf_range = 25.
    # initialize uncertainty bounds
    # pole = plane.pole
    pole_unc = fault_plane_unc
    # pole_unc = np.abs(pole * unc_bounds)
    princ_stress_vec = np.array([stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2]])
    princ_stress_unc = np.array([stress_unc[0, 0], stress_unc[1, 1], stress_unc[2, 2]])
    # princ_stress_unc = np.abs(princ_stress_vec * unc_bounds)
    #axis_unc = np.abs(axis * unc_bounds)
    # pf_guess_unc = pf * unc_bounds
    # mu_unc = np.abs(mu * unc_bounds)

    # main simulation loop
    out_data = np.empty((nsims, 3))
    for i in numba.prange(nsims):
        pole_rand = np.random.randn(3)
        pole1 = (pole_unc * pole_rand) + pole
        pole1 = pole1 / np.linalg.norm(pole1)
        # pole1 = pole_rand[i].flatten()

        stress_rand = np.random.randn(3)
        stress_rand = (princ_stress_unc * stress_rand) + princ_stress_vec

        stress1 = stress_rand * np.identity(3)

        axis_rand = np.random.randn(3)
        axis1 = (axis_unc * axis_rand) + axis
        axis1 = axis1 / np.linalg.norm(axis1)
        # axis1 = axis_rand[i].flatten()

        pf1 = np.random.random() * (pf + pf_range)
        # pf1 = pf1[0]
        mu1 = (np.random.randn() * mu_unc) + mu
        # mu1 = mu1[0]

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


#@numba.jit(forceobj=True, parallel=True)
def slip_tendency_2d(infile, inparams, mode):
    """
    Compute a deterministic 2d (i.e. where the 2d geometry is only known) slip tendency analysis. Outputs a map, as well
    as a table with the following schema:
    (ID : int) (x : float) (y : float) (Slip tendency : float) (Effective slip tendency : float)

    Parameters
    ----------
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
    # min_node_distance = 150  # 50 m
    shmax = inparams['shmax']
    shmin = inparams['shmin']
    sv = inparams['sv']
    shmaxaz = inparams['shmaxaz']
    shminaz = inparams['shminaz']
    dip = inparams['dip']
    sv_unc1 = inparams['svunc']
    az_unc = inparams['az_unc']
    shmin_unc = inparams['shmiunc']
    shmax_unc = inparams['shMunc']
    if mode == 'mc':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std = define_principal_stresses(sv, shmin, shmax, shminaz,  shmaxaz,
                                                                                    sv_unc=sv_unc1, shmin_unc=shmin_unc,
                                                                                    shmax_unc=shmax_unc, v_tilt_unc=10,
                                                                                    h_az_unc=az_unc, is_3d=False)
    elif mode == 'det':
        stress_tensor, sigma1_ax, stress_unc, sig_1_std = define_principal_stresses(sv, shmin, shmax, shminaz,
                                                                                    shmaxaz)
    else:
        raise ValueError('No recognized processing mode [''mc'' or ''det'']')
    lineaments = gp.GeoSeries.from_file(infile)
    bounds1 = lineaments.total_bounds
    #    lin_crs = lineaments.crs
    num_features = len(lineaments)
    out_features = []
    flat_out_features = []
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
            dip_rad = math.radians(dip)

            if mode == 'det':
                fault_plane = Plane(azimuth, dip_rad)
                fault_plane_pole = fault_plane.pole
                slip_tend, fail_pressure = deterministic_slip_tend(fault_plane_pole, stress_tensor, sigma1_ax,
                                                                   inparams['pf'], inparams['mu'])
                outrow = [j, work_feat_coords[j][0], work_feat_coords[j][1], work_feat_coords[j + 1][0], work_feat_coords[j + 1][1], slip_tend, fail_pressure]
            elif mode == 'mc':
                dip_unc = inparams['dipunc']
                fault_plane = Plane(azimuth, dip_rad, dip_unc=math.radians(dip_unc))
                fault_plane_pole = fault_plane.pole
                fault_plane_unc = fault_plane.pole_unc
                st_out = monte_carlo_slip_tendency(fault_plane_pole, stress_tensor, sigma1_ax, inparams['pf'],
                                                   inparams['mu'], fault_plane_unc, 0.05)
                pf_out = st_out[:, 0]
                true_data = pf_out[st_out[:, 2] > st_out[:, 1]]
                tend_out = ecdf(true_data)
                ind_50 = (np.abs(tend_out[:, 1] - 0.5)).argmin()
                outrow = [j, work_feat_coords[j][0], work_feat_coords[j][1], work_feat_coords[j + 1][0],
                          work_feat_coords[j + 1][1], tend_out[ind_50, 0]]
            else:
                raise ValueError('Cannot resolve calculation mode / not implemented yet')
            flat_out_features.append(outrow)
            fault_out_arr.append(outrow)
        out_features.append(fault_out_arr)
        print(["Finished object# ", str(i + 1)])
    return out_features, bounds1


def plot_all(out_features, flag, plot_bounds):
    if flag == 'det':
        norm1 = mpl.colors.Normalize(vmin=0.5, vmax=1.)
    elif flag == 'mc':
        norm1 = mpl.colors.Normalize(vmin=25., vmax=75.)
    for i in range(len(out_features)):
        plot_data = np.array(out_features[i])
        n_seg = int(plot_data[:, 0].max()) + 1
        segments = np.empty((n_seg, 2, 2))
        for j in range(n_seg):
            points = np.reshape(plot_data[j, 1:5], (2, 2))
            segments[j] = points
        # x1 = plot_data[:, 1].T
        # y1 = plot_data[:, 2].T
        # points = np.array([x1, y1]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if flag == 'det':
            slip_tendency = plot_data[:, 5]
        elif flag == 'mc':
            slip_tendency = plot_data[:, 5]

        lc = mpl.collections.LineCollection(segments, cmap=plt.get_cmap('jet_r'), norm=norm1)
        lc.set_array(slip_tendency)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.xlim(plot_bounds[0], plot_bounds[2])
        plt.ylim(plot_bounds[1], plot_bounds[3])
    plt.show()


if __name__ == '__main__':
    inFile_test = "./testdata/fake_lineaments.shp"
    inParams_test = {'dip': 90., 'dipunc': 10., 'shmax': 295.0, 'shMunc': 25.0, 'shmin': 77.0, 'shmiunc': 25.0,
                     'sv': 130.0, 'svunc': 25.0,
                     'depth': 5.0, 'shmaxaz': 75.0, 'shminaz': 165.0, 'azunc': 15.0, 'pf': 50.0, 'pfunc': 15.0,
                     'mu': 0.7, 'mu_unc': 0.05}
    results, bounds = slip_tendency_2d(inFile_test, inParams_test, mode='mc')
    plot_all(results, 'mc', bounds)
