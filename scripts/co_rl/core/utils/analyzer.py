import os
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, env, analyze_items, log_dir, max_episode_length=None):
        """
        Initialize the Analyzer with environment and configuration.
        """
        self.env = env
        if max_episode_length is None:
            max_episode_length = int(env.unwrapped.max_episode_length)
        self.max_episode_length = int(max_episode_length)
        # Store items to analyze (e.g. ['joint_vel', 'joint_torque'])
        self.analyze_items = analyze_items
        # Get observation info from the environment (defined in the environment's observation manager)
        self.obs_info = self.env.unwrapped.observation_manager.active_terms['obs_info']
        # Get observation configurations
        self.obs_info_cfgs = self.env.unwrapped.observation_manager._group_obs_term_cfgs['obs_info']
        # Get observation dimensions
        self.obs_info_dims = self.env.unwrapped.observation_manager._group_obs_term_dim['obs_info']
        # Articulation joint order (joint_pos / joint_vel / joint_torque obs terms).
        self.joint_names = list(self.env.unwrapped.scene["robot"].joint_names)
        # Policy action order (FL→FR→BL→BR leg groups; differs from articulation sort order).
        self.action_joint_names = self._collect_action_joint_names(self.env)
        
        # Validate analyze_items
        if not isinstance(analyze_items, list):
            raise ValueError("analyze_items should be a list of items to analyze.")

        for item in self.analyze_items:
            if item not in self.obs_info:
                raise ValueError(
                    f"[Analyzer] Invalid analyze item: '{item}' is not found in obs_info keys: {self.obs_info}"
                )
                
        # Create a directory to store analysis results
        self.log_dir = os.path.join(log_dir, "exported")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize a variable to save data  (e.g. {'joint_vel': [], 'joint_torque': []})
        self.data_store = {item: [] for item in self.analyze_items}

        self.obs_indices = {}
        idx = 0
        for key, shape in zip(self.obs_info, self.obs_info_dims):
            dim = shape[0]
            self.obs_indices[key] = (idx, idx + dim)
            idx += dim
        print(f"[Analyzer] obs_info indices: {self.obs_indices}")
        # Get observation scales for normalization
        # This will be used to scale the observations before saving
        # e.g. {'joint_pos': array([1., 1., ...]), 'joint_vel': (0.15)}
        self.obs_scales = {}
        for i, key in enumerate(self.obs_info):
            cfg = self.obs_info_cfgs[i]
            if cfg.scale is not None:
                scale_val = cfg.scale.cpu().numpy() if hasattr(cfg.scale, "cpu") else np.asarray(cfg.scale)
                scale = np.broadcast_to(scale_val, (self.obs_info_dims[i][0],)).copy()
            else:
                scale = np.ones(self.obs_info_dims[i][0])
            self.obs_scales[key] = scale
        print(f"[Analyzer] max_episode_length={self.max_episode_length}, scales={self.obs_scales}")
        if self.action_joint_names:
            print(f"[Analyzer] action_joint_names ({len(self.action_joint_names)}): {self.action_joint_names}")

        # Initialize: Create running trajectories for each environment
        # e.g. {'joint_vel': [[...], [...], [...], [...]], 'joint_torque': [[...], [...], [...], [...]]}
        self.num_envs = int(self.env.unwrapped.num_envs)
        self._running_trajectories = {
            item: [[] for _ in range(self.num_envs)]
            for item in self.analyze_items
        }

    @staticmethod
    def _collect_action_joint_names(env) -> list[str]:
        """Return joint names in policy action vector order."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        if not hasattr(unwrapped, "action_manager"):
            return []
        names: list[str] = []
        for term_name in unwrapped.action_manager.active_terms:
            term = unwrapped.action_manager.get_term(term_name)
            joint_names = getattr(term, "_joint_names", None) or getattr(term, "joint_names", None)
            if joint_names is None:
                continue
            if isinstance(joint_names, (list, tuple)):
                names.extend(joint_names)
            else:
                names.append(str(joint_names))
        return names

    def _column_headers(self, item: str, dim: int) -> list[str] | None:
        """CSV/plot column labels matched to each obs_info term layout."""
        if item == "actions":
            if len(self.action_joint_names) == dim:
                return list(self.action_joint_names)
        elif item in ("joint_pos", "joint_vel", "joint_torque"):
            if len(self.joint_names) == dim:
                return list(self.joint_names)
        if dim <= 0:
            return None
        return [f"{item}_{i}" for i in range(dim)]

    def _flush_envs(self, env_ids) -> None:
        """Move running per-env buffers into data_store."""
        for env_id in env_ids:
            for key in self.analyze_items:
                buf = self._running_trajectories[key][env_id]
                if not buf:
                    continue
                self.data_store[key].append(np.stack(buf, axis=0))
                buf.clear()

    def append(self, obs_info, dones=None):
        """Append one step of concatenated obs_info for all envs."""
        obs = obs_info.detach().cpu().numpy()
        num_envs = obs.shape[0]

        for key in self.analyze_items:
            start, end = self.obs_indices[key]
            raw = obs[:, start:end]
            scaled = raw * self.obs_scales[key]

            for env_id in range(num_envs):
                self._running_trajectories[key][env_id].append(scaled[env_id])

        if dones is not None:
            done_mask = dones.detach().cpu().numpy().reshape(-1).astype(bool)
        else:
            episode_lengths = self.env.unwrapped.episode_length_buf.cpu().numpy()
            done_mask = episode_lengths >= (self.max_episode_length - 1)

        if np.any(done_mask):
            self._flush_envs(np.where(done_mask)[0])

    def export(self):
        """Export the collected data to CSV files and plots."""
        self._flush_envs(range(self.num_envs))

        # Check if there is any data to export
        for item in self.analyze_items:
            if item not in self.data_store or len(self.data_store[item]) == 0:
                print(f"[Analyzer] Warning: No data for {item}, skipping.")
                continue

            # Concatenate the data for the item across all environments
            data = np.concatenate(self.data_store[item], axis=0)
            header = self._column_headers(item, data.shape[1])
            self._save_csv(f"{item}.csv", data, header=header)

        self._plot_all()

    def _save_csv(self, filename, data, header=None):
        """Save the data to a CSV file."""
        path = os.path.join(self.log_dir, filename)
        if header is not None:
            np.savetxt(path, data, delimiter=",", header=",".join(header), comments='')
        else:
            np.savetxt(path, data, delimiter=",")
        print(f"[Analyzer] Saved: {path}")

    def _plot_all(self):
        """Plot all the collected data."""
        data_dict = {}
        ref_dim = None

        # Check if there is any data to plot
        for key in self.analyze_items:
            if key not in self.data_store or len(self.data_store[key]) == 0:
                print(f"[Analyzer] Warning: No data for {key}, skipping.")
                continue
            # Concatenate the data for the key across all environments
            data = np.concatenate(self.data_store[key], axis=0)  # [T, D]
            if ref_dim is None:
                ref_dim = data.shape[1]
            elif data.shape[1] != ref_dim:
                raise ValueError(f"All analyze_items must have same dimension. Mismatch found in '{key}'")

            data_dict[key] = data

        # Set up the plot
        if ref_dim is not None:
            D = ref_dim
            cols, rows = 4, (D + 3) // 4
            title_source = self.analyze_items[0]
            plot_titles = self._column_headers(title_source, D) or [f"col_{i}" for i in range(D)]

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), constrained_layout=True)
            axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

            # If the items are joint torque and velocity, plot them as scatter
            is_scatter = set(self.analyze_items) == {'joint_vel', 'joint_torque'}

            for i in range(D):
                ax = axes[i]
                # Plot the data for each item
                if is_scatter:
                    x = data_dict['joint_vel'][:, i]
                    y = data_dict['joint_torque'][:, i]
                    ax.scatter(x, y, s=1, alpha=0.6)
                    ax.set_xlabel("Velocity")
                    ax.set_ylabel("Torque")
                else:
                    for key, data in data_dict.items():
                        ax.plot(np.arange(data.shape[0]), data[:, i], label=key, linewidth=1)
                    ax.set_xlabel("Time step")
                    ax.set_ylabel("Value")
                    ax.legend(fontsize=7)

                title = plot_titles[i] if i < len(plot_titles) else f"col[{i}]"
                ax.set_title(title, fontsize=9)

            for i in range(D, len(axes)):
                axes[i].axis('off')

            title_text = "Torque-Velocity Scatter" if is_scatter else "Joint-wise Comparison of Observations Over Time"
            fig.suptitle(title_text, fontsize=14)
            # Save plot to disk (headless-friendly)
            filename = "torque_velocity.png" if is_scatter else "observations_plot.png"
            path = os.path.join(self.log_dir, filename)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[Analyzer] Saved: {path}")
