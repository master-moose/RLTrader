from mmengine.registry import Registry
# from trademaster.utils import build_from_cfg
from .trademaster_utils import build_from_cfg
import copy

AGENTS = Registry('agent')
ENVIRONMENTS = Registry("environments")

def build_agent(cfg, default_args = None):
    cp_cfg = copy.deepcopy(cfg.agent)
    agent = build_from_cfg(cp_cfg, AGENTS, default_args)
    return agent