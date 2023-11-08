from typing import List, Optional, Union

import numpy as np

import yacs.config

# Default config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)

CN = Config


CONFIG_FILE_SEPARATOR = ';'

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 2023
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
            
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        # FIXME: remove later
        for i in range(len(config.CMD_TRAILING_OPTS)):
            if config.CMD_TRAILING_OPTS[i] == "DATASET.taskvars":
                if type(config.CMD_TRAILING_OPTS[i + 1]) is str:
                    config.CMD_TRAILING_OPTS[i + 1] = config.CMD_TRAILING_OPTS[i + 1].split(',')
            if config.CMD_TRAILING_OPTS[i] == 'DATASET.camera_ids':
                if type(config.CMD_TRAILING_OPTS[i + 1]) is str:
                    config.CMD_TRAILING_OPTS[i + 1] = [int(v) for v in config.CMD_TRAILING_OPTS[i + 1].split(',')]

        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
