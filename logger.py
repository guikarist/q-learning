from tqdm import tqdm


class Logger:
    _log_unit = 'step'

    def __init__(self, num_steps, log_frequency, rewards):
        self._num_steps = num_steps
        self._log_frequency = log_frequency
        self._rewards = rewards

        self._progress_bar = tqdm(total=self._num_steps, unit=Logger._log_unit)

        self._avg_reward = None
        self._max_reward = None
        if len(rewards) > 0:
            self._update_reward()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._progress_bar.close()

    def update(self, steps):
        self._progress_bar.update(steps)
        self._update_reward()

    def _update_reward(self):
        # TODO: OPTIMIZATION
        self._max_reward = max(self._rewards)
        self._avg_reward = sum(self._rewards) / len(self._rewards)
        self._update_postfix()

    def _update_postfix(self):
        self._progress_bar.set_postfix({
            'Avg_Reward': self._avg_reward,
            'Max_Reward': self._max_reward
        })
