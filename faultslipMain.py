#!/usr/bin/env python3


import os
import sys
import math
import numpy as np
import matplotlib as mpl
import geopandas as gp
import shapely as sp


def rot_matrix_axis_angle(axis, angle):
    a1 = axis[0]
    a2 = axis[1]
    a3 = axis[3]
    #    rot = math.radians(angle)
    ident = np.identity(3)
    out = np.outer(axis, axis)
    a_x = np.array([[0, -a3, a2], [a3, 0, -a2], [-a2, a1, 0]])
    rot_matrix = (math.cos(angle) * ident) + (math.sin(angle) * a_x) + (1 - math.cos(angle) * out)
    return rot_matrix


def define_principal_stresses(sv, shmin, shmax, hminaz, hmaxaz):
    # generates cauchy stress tensor from principal stress directions, assumes vertical stress in one direction
    # rotates sigma1 (maximum principal stress) direction to plane normal
    if abs(hmaxaz - hminaz) != 90:
        raise ValueError('hmin and hmax are not orthogonal')
    #    rotangle = math.radians(hmaxaz)
    if sv > shmin & sv > shmax:
        sigma1 = sv
        sigma2 = shmax
        sigma3 = shmin
        az_rad = 0
        az_dip_rad = math.pi / 2
        rotated_axis = [math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                        math.sin(az_dip_rad)]
    elif sv < shmax & sv > shmin:
        sigma1 = shmax
        sigma2 = sv
        sigma3 = shmin
        az_rad = math.radians(shmax)
        az_dip_rad = 0
        rotated_axis = [math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                        math.sin(az_dip_rad)]
    elif sv < shmax & sv < shmin:
        sigma1 = shmax
        sigma2 = shmin
        sigma3 = sv
        az_rad = math.radians(shmax)
        az_dip_rad = 0
        rotated_axis = [math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                        math.sin(az_dip_rad)]
    else:
        raise ValueError('Unable to differentiate principal stress directions')
    princ_stress_tensor = np.array([[sigma1, 0, 0], [0, sigma2, 0], [0, 0, sigma3]])
    return princ_stress_tensor, rotated_axis


def rotate_plane_stress(sigma_1_ax, plane_norm, princ_stress_tensor):
    rot_axis = np.cross(plane_norm, sigma_1_ax)
    rot_angle = -1 * np.dot(plane_norm, sigma_1_ax)
    rotmatrix = rot_matrix_axis_angle(rot_axis, rot_angle)
    plane_stress_tensor = rotmatrix.T @ princ_stress_tensor @ rotmatrix
    return plane_stress_tensor


def point_az(p1, p2):
    x1 = p1(0)
    x2 = p2(0)
    y1 = p1(1)
    y2 = p2(1)
    azimuth1 = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
    azimuth = (azimuth1 + 360) % 360
    return azimuth


def redistribute_vertices(geom, distance):
    from shapely.geometry import LineString
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def det_2d_slip_tendency(inFile, inParams):
    node_length = 50  # 50 m
    shmax = inParams['shmax']
    shmin = inParams['shmin']
    sv = inParams['sv']
    shmaxaz = inParams['shmaxaz']
    shminaz = inParams['shminaz']
    dip = inParams['dip']
    stress_tensor, sigma1_ax = define_principal_stresses(sv, shmin, shmax, shminaz, shmaxaz)
    lineaments = gp.GeoSeries.from_file(inFile)
    #    lin_crs = lineaments.crs
    num_features = len(lineaments)
    out_features = []
    for i in range(num_features):
        work_feat = lineaments[i]
        work_feat_interp = redistribute_vertices(work_feat, node_length)
        num_nodes = len(work_feat_interp)
        fault_out_arr = np.zeros((num_nodes,5))
        for j in range(1, num_nodes - 1):
            point1 = work_feat_interp[j - 1]
            point2 = work_feat_interp[j + 1]
            azimuth = point_az(point1, point2)
            dip_dir = azimuth + 90
            az_rad = math.radians(azimuth)
            az_dip_rad = 0
            dip_dir_az = math.radians(dip_dir)
            dip_rad = math.radians(dip)
            v1 = [math.sin(az_rad) * math.cos(az_dip_rad), math.cos(az_rad) * math.cos(az_dip_rad),
                  math.sin(az_dip_rad)]
            v2 = [math.sin(dip_dir_az) * math.cos(dip_rad), math.cos(dip_dir_az) * math.cos(dip_rad),
                  math.sin(dip_rad)]
            fault_plane_pole = np.cross(v1, v2)
            plane_stress = rotate_plane_stress(sigma1_ax, fault_plane_pole, stress_tensor)
            sigma_n = plane_stress[0, 0]
            sigma_t = plane_stress[2, 0]
            slip_tendency = sigma_t / sigma_n
            sigma_n_eff = sigma_n - InParams['pf']
            slip_tendency_eff = sigma_t / sigma_n_eff
            fault_out_arr[j-1,0] = j
            fault_out_arr[j-1,1] = point1
            fault_out_arr[j-1,2] = point2
            fault_out_arr[j-1,3] = slip_tendency
            fault_out_arr[j-1,4] = slip_tendency_eff
        out_features.append(fault_out_arr)
    return out_features





if __name__ == '__main__':
    inFile_test = "./testdata/fake_lineaments.shp"
    inParams_test = {'dip': 60, 'dipunc': 10, 'shmax': 295.0, 'shMunc': 25.0, 'shmin': 77.0, 'shmiunc': 25.0,
                     'sv': 130.0, 'svunc': 25.0,
                     'depth': 5.0, 'shmaxaz': 75.0, 'shminaz': 165.0, 'azunc': 15.0, 'pf': 50.0, 'pfunc': 15.0, 'mu' : 0.5, 'mu_unc' : 0.05}
    results = det_2d_slip_tendency(inFile_test, inParams_test)
