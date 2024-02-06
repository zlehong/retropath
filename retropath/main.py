import json
import random
import logging
from pathlib import Path
from typing import Literal, Union, IO, Optional

import numpy as np
import pandas as pd
from ladybug.epw import EPW
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import rotate

from utils.energy import ShoeboxConfiguration

SHOEBOX_WIDTH = 3.5
PERIM_DEPTH = 4.5

logging.basicConfig()
logger = logging.getLogger("RETROPATH")
logger.setLevel(logging.INFO)


class RetroPathBuilding:
    """
    Building object to be retrofitted.
    3D model definition: geopandas dataframe [floor polygon, height, program]
    """

    def __init__(
        self,
        name,
        building_gdf: gpd.GeoDataFrame,
        building_config: pd.DataFrame,
        epw: EPW,
        **kwargs,
    ):
        self.name = name
        self.building_gdf = building_gdf
        self.building_config = building_config
        self.epw = epw

    def build_shoebox_df(self):  # TODO: method
        """
        allocate shoeboxes with given method, merge with building config
        """
        pass

    def autozone(self):
        pass

    def allocate_shoeboxes(self):
        pass

    def init_energy_model(self, type: Literal["eplus", "local", "api"] = "local"):
        pass

    def run_energy_model(self):
        pass

    def get_retrofit_options(self):
        pass

    def plan_retrofit_path(self):
        pass


def random_building(
    epw_path, config_path, n_programs=2, n_floors=2, length_range=[10, 70]
):
    # Load a random config
    with open(config_path) as f:
        config = json.load(f)
    building_configs = {}
    for i in range(n_programs):
        building_config = {}
        for param, param_def in config.items():
            if param != "height":
                if param_def["mode"] == "Onehot":
                    building_config[param] = random.randint(
                        0, param_def["option_count"]
                    )
                elif param_def["mode"] == "Continuous":
                    building_config[param] = random.uniform(
                        param_def["min"], param_def["max"]
                    )
                else:
                    print(f"Error: {param} has unknown type {param_def['mode']}")
        building_config["width"] = SHOEBOX_WIDTH
        building_configs[f"program_{i}"] = building_config

    # Initiate a new random building shape
    building_config = pd.DataFrame.from_dict(building_configs, orient="index")
    # Sample random rectangle
    length = random.uniform(length_range[0], length_range[1])
    width = random.uniform(length_range[0], length_range[1])
    rectangle = Polygon([(0, 0), (length, 0), (length, width), (0, width)])

    # Rotate randomly
    orientation = random.uniform(0, 2 * np.pi)
    rectangle = rotate(rectangle, orientation, use_radians=True)
    # Copy n_floors times (reshape/scale?)
    building_gdf = gpd.GeoDataFrame(geometry=[rectangle] * n_floors)
    # Assign random heights
    building_gdf["height"] = np.random.uniform(
        config["height"]["min"], config["height"]["max"], size=(building_gdf.shape[0],)
    )
    # Assign random program
    programs = list(building_configs.keys())
    idxs = random.sample(range(n_floors), building_gdf.shape[0])
    building_gdf["program"] = [programs[x] for x in idxs]

    return RetroPathBuilding(
        name="test",
        building_config=building_config,
        building_gdf=building_gdf,
        epw=EPW(epw_path),
    )


if __name__ == "__main__":
    epw_path = "retropath/data/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw"
    config_path = "retropath/data/space_definition.json"
    bldg = random_building(epw_path, config_path)
    print(bldg.building_gdf)
    print(bldg.building_config)
    print(bldg.epw)

    # Run baseline energy models
