import os
import json
from uuid import UUID, uuid4
import requests
import logging
from pathlib import Path
from typing import Literal, Union, IO, Optional
import sys

from lightning.pytorch import Trainer

from ladybug.epw import EPW
import pandas as pd

BACKEND_URL = os.getenv("BACKEND_URL")
SHADING_DIV_SIZE = 12

logging.basicConfig()
logger = logging.getLogger("ENERGY")
logger.setLevel(logging.INFO)


class ShoeboxConfiguration:
    """
    Stateful class for shoebox object args
    """

    __slots__ = (
        "width",
        "height",
        "perim_depth",
        "core_depth",
        # "adiabatic_partition_flag",
        "roof_2_footprint",
        "ground_2_footprint",
        "wwr",
        "orientation",
        "shading_vect",
    )

    def __init__(self):
        """
        Builder throws error if core is less than 2.
        Set adiabatic partition flag. Check that this works - adiabatic wall & various core depths.
        """
        self.shading_vect = np.zeros(SHADING_DIV_SIZE)


class EnergyHandler:
    """
    Handles a dataframe shoebox feature end-use modeling with EnergyPlus, ML-local, or API
    """

    def __init__(
        self,
        shoebox_features: pd.DataFrame,
        type: Literal["eplus", "local", "api"] = "local",
        eplus_path: Optional[Union[str, IO, Path]] = None,
        local_path: Optional[Union[str, IO, Path]] = None,
    ):
        self.shoebox_features = shoebox_features
        self.emodel = self.init_energy_model(type)

    def init_energy_model(self, type):
        if type == "eplus":
            pass
        elif type == "local":
            emodel = energySurrogate()
        elif type == "api":
            pass
        return emodel


class energySurrogate:
    def __init__(
        self,
        local_loc="C:/Users/zoele/Git_Repos/ml-for-building-energy-modeling/ml-for-bem",
    ):
        sys.path.append(local_loc)
        from ml.surrogate import Surrogate
        from ml.predict import predict_ubem

    def load_surrogate(
        self,
        registry="ml-for-building-energy-modeling/model-registry",
        model="Global UBEM Shoebox Surrogate with Combined TS Embedder",
        tag="v3",
        resource="model.ckpt",
    ):
        surrogate = Surrogate.load_from_registry(registry, model, tag, resource)
        surrogate.model.eval()
        torch.set_float32_matmul_precision("medium")

        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            enable_progress_bar=True,
        )


class energyAPI:
    def __init__(self):
        pass

    def submit_job(template_config, scheds, epw: EPW):
        # TODO check inputs
        url = f"{BACKEND_URL}/building"
        uuid = uuid4()
        tmp = f"data/temp/frontend/{uuid}"
        os.makedirs(tmp, exist_ok=True)
        epw.save(f"{tmp}/epw.epw")
        template_data = {
            "features": template_config,
            "schedules": scheds.tolist(),
        }
        with open(f"{tmp}/template.json", "w") as f:
            json.dump(template_data, f)
        # TODO: template file should be in body or query params
        files = {
            "epw_file": open(f"{tmp}/epw.epw", "rb"),
            "template_file": open(f"{tmp}/template.json", "rb"),
        }
        query_params = {}
        query_params["uuid"] = uuid

        response = requests.post(
            url,
            files=files,
            params=query_params,
        )
        if response.status_code == 200:
            run_results = {}
            run_data = response.json()
            annual = pd.DataFrame.from_dict(run_data["annual"], orient="tight")
            monthly = pd.DataFrame.from_dict(run_data["monthly"], orient="tight")
            shoeboxes = pd.DataFrame.from_dict(run_data["shoeboxes"], orient="tight")
            run_results["annual"] = annual
            run_results["monthly"] = monthly
            run_results["shoeboxes"] = shoeboxes
            logger.info("Success!")
        else:
            logger.info(f"Error submitting job. Please try again.")
