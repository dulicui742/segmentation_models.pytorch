from typing import Callable
# from input_config import transforms, time, tfmt, os
import json
import itertools
import pdb
# import custom_transforms
import torch
import glom
import time
import os


# new class mimic input_config.py's Config
class ConfigBase:
    def __getattr__(self, key):
        value = glom.glom(self.__kv_dict, key)
        if isinstance(value, dict):
            return ConfigBase(kv_dict=value)
        return value

    def __len__(self):
        return self.__kv_dict.__len__()

    def __init__(self, *, kv_dict):
        self.__kv_dict = kv_dict

    def items(self):
        return self.__kv_dict.items()

    def to_dict(self):
        return self.__kv_dict

    def report(self):
        good_keys = []
        pool = dict(itertools.chain(self.__kv_dict.items(), self.__dict__.items()))
        for key, value in pool.items():
            # skip hidden fields
            if key.startswith("_"):
                continue
            # try to dump everything
            try:
                json.dumps(value)
            except:
                print(f"Skip {key}")
                continue
            else:
                good_keys.append(key)
        return json.dumps({k: v for k, v in pool.items() if k in good_keys}, indent=2)

    @classmethod
    def init_from_env(cls, **kwargs):
        env_file = os.environ["USE_JSON_CONFIG_FILE"]
        with open(env_file) as IN:
            info = json.load(IN)
        opt = cls(**{"kv_dict": info, **kwargs})
        # report
        print("env_file: ", env_file)
        report_file = env_file + ".output_{}.json" .format(opt.timestamp)
        with open(report_file, "w") as OUT:
            print(opt.report(), file=OUT)
        opt.report_file = report_file
        return opt


class Config(ConfigBase):
    tfmt = "%m%d_%H%M%S"
    def __init__(self, *, kv_dict):
        super().__init__(kv_dict=kv_dict)
        print(self.tfmt)
        kv_dict["timestamp"] = (
            time.strftime(self.tfmt)
        )
        # all magics
        self.__kv_dict = kv_dict

    @classmethod
    def init_from_env(cls, *, ask_for_continue=True):
        opt = super().init_from_env()
        if ask_for_continue:
            assert (
                input(f"Report file({opt.report_file}) created. Continue? ").lower() == "y"
            )
        return opt


class InferenceConfig(ConfigBase):

    def __init__(self, *, kv_dict):
        super().__init__(kv_dict=kv_dict)

