from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any

class BaseActioner:

    def reset(self, task_str, variation, instructions, demo_id):
        self.task_str = task_str
        self.variation = variation
        self.instructions = instructions
        self.demo_id = demo_id

        self.step_id = 0
        self.state_dict = {}
        self.history_obs = {}

    def predict(self, *args, **kwargs):
        raise NotImplementedError('implete predict function')
