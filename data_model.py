import dataclasses
import math
import json
import numpy as np


@dataclasses.dataclass
class UncClass:
    _default_unc = 0.15  # 15%
    mean: float
    unit_name: str = ""
    std_perc: float = _default_unc

    def std_unit(self):
        return self.mean * self.std_perc

    def std_perc_100(self):
        return self.std_perc * 100.

    def perc_100_to_perc(self, perc_100):
        if perc_100 < 1:
            raise ValueError("Input percentage needs to be greater than 1")
        self.std_perc = perc_100 / 100.

    def unit_to_perc(self, unit_std):
        self.std_perc = unit_std / self.mean


class ModelDefaults:
    def __init__(self):
        infile = "./defaults.json"
        with open(infile) as load_file:
            j_data = json.load(load_file)
        inputs = j_data['input_data'][0]
        self.general_unc = inputs["general_unc"]
        self.max_pf = inputs['pf']
        self.pf_unit = "MPa"
        self.density = inputs["density"]
        self.density_unit = "kg/m^3"
        self.hydro = inputs["hydro"]
        self.hydro_unit = "MPa/km"
        self.dip = inputs["dip"]
        self.dip_unit = "deg"
        az_unc = inputs["az_unc"]
        self.az_unit = "deg"
        self.az_unc_perc = az_unc / 360.
        self.sv = (self.density * 9.81) / 1000  # MPa/km
        self.stress_unit = "MPa/km"
        self.sh_max_az = inputs["sh_max_az"]
        self.sh_min_az = inputs["sh_min_az"]
        self.mu = inputs["mu"]

        self.F_mu = (math.sqrt(self.mu ** 2 + 1)) ** 2
        abs_shmax = self.F_mu * (self.sv - self.hydro) + self.hydro
        abs_shmin = ((self.sv - self.hydro) / self.F_mu) + self.hydro
        self.shmax_r = abs_shmax
        self.shmin_r = (abs_shmax - self.sv) / 2
        self.shmax_ss = abs_shmax
        self.shmin_ss = abs_shmin
        self.shmax_n = (self.sv - abs_shmin) / 2
        self.shmin_n = abs_shmin


class ModelInputs:
    def __init__(self, input_dict):
        """

          Parameters
          ----------
          input_dict
          """
        defaults = ModelDefaults()
        if "max_pf" in input_dict.values():
            self.max_pf = input_dict["max_pf"]
        else:
            self.max_pf = defaults.max_pf

        if "dip" in input_dict.values():
            if "dip_unc" in input_dict.values():
                self.Dip = UncClass(input_dict["dip"], defaults.dip_unit, input_dict["dip_unc"])
            else:
                self.Dip = UncClass(input_dict["dip"], defaults.dip_unit)
        else:
            self.Dip = UncClass(defaults.dip, defaults.dip_unit)
        if "sv" in input_dict.values():
            if "sv_unc" in input_dict.values():
                self.Sv = UncClass(input_dict["sv"], defaults.stress_unit, input_dict["sv_unc"])
            else:
                self.Sv = UncClass(input_dict["sv"], defaults.stress_unit)
        else:
            self.Sv = UncClass(defaults.sv, defaults.stress_unit)
        if "s_hydro" in input_dict.values():
            self.SHydro = UncClass(input_dict["s_hydro"], defaults.stress_unit)
        else:
            self.SHydro = UncClass(defaults.hydro, defaults.stress_unit)
        if "mu" in input_dict.values():
            if "mu_unc" in input_dict.values():
                self.Mu = UncClass(input_dict["mu"], input_dict["mu_unc"])
            else:
                self.Mu = UncClass(defaults.mu)
        else:
            self.Mu = UncClass(defaults.mu)

        if "shmax" in input_dict.values():
            shmax = float(input_dict['shmax'])
            if "shmin" in input_dict.values():
                shmin = float(input_dict['shmin'])
                if shmax > self.Sv.mean > shmin:
                    self.ShMaxSS = UncClass(shmax, defaults.stress_unit)
                    self.ShMinSS = UncClass(shmin, defaults.stress_unit)
                    self.ShMaxR = UncClass(defaults.shmax_r, defaults.stress_unit)
                    self.ShMinR = UncClass(defaults.shmin_r, defaults.stress_unit)
                    self.ShMaxN = UncClass(defaults.shmax_n, defaults.stress_unit)
                    self.ShMinN = UncClass(defaults.shmin_n, defaults.stress_unit)
                elif shmax > shmin > self.Sv.mean:
                    self.ShMaxR = UncClass(shmax, defaults.stress_unit)
                    self.ShMinR = UncClass(shmin, defaults.stress_unit)
                    self.ShMaxSS = UncClass(defaults.shmax_ss, defaults.stress_unit)
                    self.ShMinSS = UncClass(defaults.shmin_ss, defaults.stress_unit)
                    self.ShMaxN = UncClass(defaults.shmax_n, defaults.stress_unit)
                    self.ShMinN = UncClass(defaults.shmin_n, defaults.stress_unit)
                elif self.Sv.mean > shmax > shmin:
                    self.ShMaxN = UncClass(shmax, defaults.stress_unit)
                    self.ShMinN = UncClass(shmin, defaults.stress_unit)
                    self.ShMaxR = UncClass(defaults.shmax_r, defaults.stress_unit)
                    self.ShMinR = UncClass(defaults.shmin_r, defaults.stress_unit)
                    self.ShMaxSS = UncClass(defaults.shmax_ss, defaults.stress_unit)
                    self.ShMinSS = UncClass(defaults.shmin_ss, defaults.stress_unit)
        else:
            # print("default")
            self.ShMaxR = UncClass(defaults.shmax_r, defaults.stress_unit)
            self.ShMinR = UncClass(defaults.shmin_r, defaults.stress_unit)
            self.ShMaxSS = UncClass(defaults.shmax_ss, defaults.stress_unit)
            self.ShMinSS = UncClass(defaults.shmin_ss, defaults.stress_unit)
            self.ShMaxN = UncClass(defaults.shmax_n, defaults.stress_unit)
            self.ShMinN = UncClass(defaults.shmin_n, defaults.stress_unit)

        if "shmaxaz" in input_dict.values():
            if "az_unc" in input_dict.values():
                self.ShMaxAz = UncClass(input_dict["shmaxaz"], defaults.az_unit, input_dict["az_unc"])
            else:
                self.ShMaxAz = UncClass(input_dict["shmaxaz"], defaults.az_unit, defaults.az_unc_perc)
        else:
            self.ShMaxAz = UncClass(defaults.sh_max_az, defaults.az_unit, defaults.az_unc_perc)
        if "shminaz" in input_dict.values():
            if "az_unc" in input_dict.values():
                self.ShMinAz = UncClass(input_dict["shminaz"], defaults.az_unit, input_dict["az_unc"])
            else:
                self.ShMinAz = UncClass(input_dict["shminaz"], defaults.az_unit, defaults.az_unc_perc)
        else:
            if "shmaxaz" in input_dict.values():
                self.ShMinAz = UncClass(self.ShMaxAz.mean + 90., defaults.az_unit, defaults.az_unc_perc)
            else:
                self.ShMinAz = UncClass(defaults.sh_min_az, defaults.az_unit, defaults.az_unc_perc)


class SegmentDet2dResult:
    """

    """
    def __init__(self, x1, y1, x2, y2, result, metadata):
        """"""
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        # self.pf_results = pf_results
        if "line_id" in metadata:
            self.line_id = metadata["line_id"]
        if "seg_id" in metadata:
            self.seg_id = metadata["seg_id"]
        self.result = result


class MeshFaceResult:
    """

    """
    def __init__(self, face_num, triangle, p1, p2, p3, pf_results):
        """

        Parameters
        ----------
        face_num
        p1
        p2
        p3
        """
        self.face_num = face_num
        self.triangle = triangle
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        if pf_results.size == 0:
            x = np.array([0., 0., 0.])
            y = np.array([0., 0.5, 1.])
            self.ecdf = np.column_stack((x, y))
        pf_results.sort()
        n = pf_results.size
        y = np.linspace(1.0 / n, 1, n)
        self.ecdf = np.column_stack((pf_results, y))

    def ecdf_cutoff(self, cutoff):
        """

        Parameters
        ----------
        cutoff: float

        Returns
        -------

        """
        # self.ecdf[:, 0] = self.ecdf[:, 0] - hydrostatic_pres
        ind_fail = (np.abs(self.ecdf[:, 1] - cutoff)).argmin()
        fail_pressure = self.ecdf[ind_fail, 0]
        return fail_pressure


class SegmentMC2dResult:
    """

    """
    def __init__(self, x1, y1, x2, y2, pf_results, metadata):

        """"""
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        # self.pf_results = pf_results
        if "line_id" in metadata:
            self.line_id = metadata["line_id"]
        if "seg_id" in metadata:
            self.seg_id = metadata["seg_id"]

        if pf_results.size == 0:
            x = np.array([0., 0., 0.])
            y = np.array([0., 0.5, 1.])
            self.ecdf = np.column_stack((x, y))
        pf_results.sort()
        n = pf_results.size
        y = np.linspace(1.0 / n, 1, n)
        self.ecdf = np.column_stack((pf_results, y))

    def ecdf_cutoff(self, cutoff):
        """

        Parameters
        ----------
        cutoff: float

        Returns
        -------

        """
        # self.ecdf[:, 0] = self.ecdf[:, 0] - hydrostatic_pres
        ind_fail = (np.abs(self.ecdf[:, 1] - cutoff)).argmin()
        fail_pressure = self.ecdf[ind_fail, 0]
        return fail_pressure


class Results2D:
    """
    Class to manage 2D results.
    """

    def __init__(self, input_list, **kwargs):
        """

        Parameters
        ----------
        input_list
        """
        # self.segment_list = input_list
        num_lines = len(np.unique([obj.line_id for obj in input_list]))
        self.lines = []
        for line in range(num_lines):
            line_list = [obj for obj in input_list if obj.line_id == line]
            line_list.sort(key=lambda obj: obj.seg_id, reverse=False)
            self.lines.append(line_list)

        if "cutoff" in kwargs:
            self.cutoff = float(kwargs["cutoff"])
        if "ecdf" in input_list[0].__dict__:
            self.type = "mc"
        elif "result" in input_list[0].__dict__:
            self.type = "det"
        x = []
        y = []
        result = []
        for obj in input_list:
            x.append(obj.p1[0])
            x.append(obj.p2[0])
            y.append(obj.p1[1])
            y.append(obj.p2[1])
            if self.type == "mc":
                result.append(obj.ecdf_cutoff(self.cutoff))
            if self.type == "det":
                result.append(obj.result)
        self.xmin = min(x)
        self.xmax = max(x)
        self.ymin = min(y)
        self.ymax = max(y)
        self.plotmin = min(result)
        self.plotmax = max(result)


    def update_cutoff(self, cutoff):
        """

        Returns
        -------

        """
        if self.type == "det":
            return
        result = []
        for line in self.lines:
            for obj in line:
                result.append(obj.ecdf_cutoff(cutoff))
        self.plotmin = min(result)
        self.plotmax = max(result)
        return


