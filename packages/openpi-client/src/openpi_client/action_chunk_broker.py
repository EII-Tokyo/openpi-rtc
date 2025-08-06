from typing import Dict

import json
import pathlib
import threading
import time
import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, is_rtc: bool = False):
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_origin_actions: np.ndarray | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False

        self._obs: Dict[str, np.ndarray] | None = None
        self._s = 25
        self._d = 10
        self._is_rtc = is_rtc
        self._norm_stats = json.loads(pathlib.Path("/app/checkpoints/19999/assets/trossen/norm_stats.json").read_text())["norm_stats"]

        if self._is_rtc:
            self._infer_thread = threading.Thread(target=self._background_infer)
            self._infer_thread.start()

    def _background_infer(self):
        while True:
            if self._cur_step == self._s:
                # start_time = time.time()
                self._background_running = True

                # flip, normalize joint actions
                norm_action = (np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._last_results["actions"] - np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * self._obs["state"][:14] - np.array(self._norm_stats["actions"]["mean"])[:14]) / (np.array(self._norm_stats["actions"]["std"])[:14] + 1e-6)
                
                # get normalized gripper action
                norm_action[:, 6] = self._last_origin_actions[:, 6]
                norm_action[:, 13] = self._last_origin_actions[:, 13]
                
                zeros_padding = np.zeros((norm_action.shape[0], 18))
                norm_action = np.concatenate([norm_action, zeros_padding], axis=1)
                self._background_results = self._policy.infer(self._obs, norm_action, self._is_rtc)

                self._background_running = False
                # end_time = time.time()
                # print(f"Time taken to infer: {end_time - start_time}")
            else:
                time.sleep(0.01)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006 
        if self._is_rtc:
            # init     
            if self._last_results is None:
                self._last_results = self._policy.infer(obs, None, self._is_rtc)
                self._last_origin_actions = self._last_results["origin_actions"]
                self._last_state = self._last_results["state"]
                self._last_results = {"actions": self._last_results["actions"]}
                self._cur_step = 0

            results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            self._obs = obs
            self._cur_step += 1

            # if current step equals s+d, wait for background inference to complete
            if self._cur_step == self._s + self._d:
                while self._background_running:
                    time.sleep(0.01)
                self._last_origin_actions = self._background_results["origin_actions"]
                self._last_state = self._background_results["state"]
                self._last_results = {"actions": self._background_results["actions"]}
                self._cur_step -= self._s

            return results

        else:
            if self._last_results is None:
                self._last_results = self._policy.infer(obs)
                self._cur_step = 0
            
            self._last_results = {"actions": self._last_results["actions"]}

            results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                self._last_results = None

            return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
