import dataclasses
import math
import json


@dataclasses.dataclass
class UncClass:
    _default_unc = 0.15  # 15%
    mean: float
    std_perc: float = _default_unc

    def std_unit(self):
        return self.mean * self.std_perc


class ModelDefaults:
    def __init__(self):
        infile = "./defaults.json"
        with open(infile) as load_file:
            j_data = json.load(load_file)
        inputs = j_data['input_data'][0]
        self.general_unc = inputs["general_unc"]
        self.density = inputs["density"]
        self.hydro = inputs["hydro"]
        self.dip = inputs["dip"]
        az_unc = inputs["az_unc"]
        self.az_unc_perc = az_unc / 360.
        self.sv = (self.density * 9.81) / 1e6  # MPa/km
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

        if "dip" in input_dict.values():
            if "dip_unc" in input_dict.values():
                self.Dip = UncClass(input_dict["dip"], input_dict["dip_unc"])
            else:
                self.Dip = UncClass(input_dict["dip"])
        else:
            self.Dip = UncClass(defaults.dip)
        if "sv" in input_dict.values():
            if "sv_unc" in input_dict.values():
                self.Sv = UncClass(input_dict["sv"], input_dict["sv_unc"])
            else:
                self.Sv = UncClass(input_dict["sv"])
        else:
            self.Sv = UncClass(defaults.sv)
        if "s_hydro" in input_dict.values():
            self.SHydro = UncClass(input_dict["s_hydro"])
        else:
            self.s_hydro = UncClass(defaults.hydro)
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
                    self.ShMaxSS = shmax
                    self.ShMinSS = shmin
                    self.ShMaxR = defaults.shmax_r
                    self.ShMinR = defaults.shmin_r
                    self.ShMaxN = defaults.shmax_n
                    self.ShMinN = defaults.shmin_n
                elif shmax > shmin > self.Sv.mean:
                    self.ShMaxR = shmax
                    self.ShMinR = shmin
                    self.ShMaxSS = defaults.shmax_ss
                    self.ShMinSS = defaults.shmin_ss
                    self.ShMaxN = defaults.shmax_n
                    self.ShMinN = defaults.shmin_n
                elif self.Sv.mean > shmax > shmin:
                    self.ShMaxN = shmax
                    self.ShMinN = shmin
                    self.ShMaxR = defaults.shmax_r
                    self.ShMinR = defaults.shmin_r
                    self.ShMaxSS = defaults.shmax_ss
                    self.ShMinSS = defaults.shmin_ss
        else:
            self.ShMaxR = defaults.shmax_r
            self.ShMinR = defaults.shmin_r
            self.ShMaxSS = defaults.shmax_ss
            self.ShMinSS = defaults.shmin_ss
            self.ShMaxN = defaults.shmax_n
            self.ShMinN = defaults.shmin_n

        if "shmaxaz" in input_dict.values():
            if "az_unc" in input_dict.values():
                self.ShMaxAz = UncClass(input_dict["shmaxaz"], input_dict["az_unc"])
            else:
                self.ShMaxAz = UncClass(input_dict["shmaxaz"], defaults.az_unc_perc)
        else:
            self.ShMaxAz = UncClass(defaults.sh_max_az, defaults.az_unc_perc)
        if "shminaz" in input_dict.values():
            if "az_unc" in input_dict.values():
                self.ShMinAz = UncClass(input_dict["shminaz"], input_dict["az_unc"])
            else:
                self.ShMinAz = UncClass(input_dict["shminaz"], defaults.az_unc_perc)
        else:
            if "shmaxaz" in input_dict.values():
                self.ShMinAz = UncClass(self.ShMaxAz.mean + 90., defaults.az_unc_perc)
            else:
                self.ShMinAz = UncClass(defaults.sh_min_az, defaults.az_unc_perc)
