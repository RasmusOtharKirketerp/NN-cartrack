# hdf5_logger.py
import h5py
import numpy as np

class HDF5Logger:
    def __init__(self, filename="training_logs.hdf5"):
        """
        Initialize the HDF5 logger.
        :param filename: Name of the HDF5 file.
        """
        self.filename = filename

    def save_logs(self, log_entries):
        """
        Save structured log data to an HDF5 file.
        :param log_entries: List of log dictionaries.
        """
        with h5py.File(self.filename, 'w') as f:
            # Store top-level metrics as datasets
            f.create_dataset('episode', data=np.array([entry['episode'] for entry in log_entries]))
            f.create_dataset('reward', data=np.array([entry['reward'] for entry in log_entries]))
            f.create_dataset('avg_reward_100', data=np.array([entry['avg_reward_100'] for entry in log_entries]))
            f.create_dataset('epsilon', data=np.array([entry['epsilon'] for entry in log_entries]))
            f.create_dataset('avg_loss', data=np.array([entry['avg_loss'] for entry in log_entries]))
            f.create_dataset('max_q_value', data=np.array([entry['max_q_value'] for entry in log_entries]))
            f.create_dataset('steps', data=np.array([entry['steps'] for entry in log_entries]))

            # Create a group for reward breakdowns
            breakdown_group = f.create_group('reward_breakdown')
            breakdown_group.create_dataset('delta_x_reward', data=np.array([entry['reward_breakdown']['delta_x_reward'] for entry in log_entries]))
            breakdown_group.create_dataset('border_penalty', data=np.array([entry['reward_breakdown']['border_penalty'] for entry in log_entries]))
            breakdown_group.create_dataset('obstacle_penalty', data=np.array([entry['reward_breakdown']['obstacle_penalty'] for entry in log_entries]))

    def load_logs(self):
        """
        Load structured log data from an HDF5 file.
        :return: List of log dictionaries.
        """
        with h5py.File(self.filename, 'r') as f:
            logs = []
            for i in range(len(f['episode'])):
                logs.append({
                    "episode": f['episode'][i],
                    "reward": f['reward'][i],
                    "avg_reward_100": f['avg_reward_100'][i],
                    "epsilon": f['epsilon'][i],
                    "avg_loss": f['avg_loss'][i],
                    "max_q_value": f['max_q_value'][i],
                    "steps": f['steps'][i],
                    "reward_breakdown": {
                        "delta_x_reward": f['reward_breakdown/delta_x_reward'][i],
                        "border_penalty": f['reward_breakdown/border_penalty'][i],
                        "obstacle_penalty": f['reward_breakdown/obstacle_penalty'][i],
                    }
                })
            return logs

    def append_log(self, log_entry):
        """
        Append a single log entry to the HDF5 file (not implemented due to HDF5 limitations).
        Recommendation: Collect logs in memory and save them in bulk using `save_logs`.
        """
        raise NotImplementedError("HDF5 does not support efficient appending. Use save_logs to write in bulk.")
