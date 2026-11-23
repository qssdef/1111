from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EnvConfig:
    map_size: float = 100.0
    n_pursuers: int = 6
    n_evaders: int = 2
    dt: float = 0.25
    max_steps: int = 150
    pursuer_max_speed: float = 2.0
    evader_max_speed: float = 2.2
    max_acceleration: float = 0.4
    max_turn_rate: float = 0.35  # rad/s
    current_max_speed: float = 0.3
    current_grid_size: int = 5
    obstacle_count: int = 2
    collision_failure_steps: int = 0  # disabled: collisions do not retire groups
    obstacle_radius_range: Tuple[float, float] = (2.0, 5.0)
    safety_margin: float = 1.5
    capture_radius: float = 5.0
    capture_min_agents: int = 3
    capture_min_angle_gap: float = 0.40
    capture_max_angle_gap: float = 3.10  # < pi: target still lies inside the enclosing polygon
    capture_safe_distance: float = 1.5
    capture_hold_steps: int = 4  # 1 second at dt=0.25; interruptions reset the count
    reassign_interval: int = 4
    assignment_switch_cost: float = 8.0
    assignment_hysteresis: float = 2.0
    cross_probability_per_second: float = 0.18
    cross_duration: float = 24.0
    cross_min_start_distance: float = 10.0
    cross_max_start_distance: float = 50.0
    cross_encounter_distance: float = 6.0
    cross_safe_distance: float = 2.0
    cross_lane_offset: float = 1.75
    cross_path_lookahead: float = 4.0
    cross_exit_distance: float = 12.0
    cross_boundary_margin: float = 4.0
    cross_cooldown: float = 5.0
    spawn_margin: float = 12.0
    max_episode_steps: int = 150
    pursuit_observations: bool = False
    pursuit_spawn_radius_range: Tuple[float, float] = (0.0, 0.0)
    pursuit_spawn_sector: float = 2.6
    assignment_mode: str = "fixed"
    individual_pursuer_rewards: bool = False
    evader_speed_limit: Optional[float] = None  # curriculum motion limit; observation scale stays unchanged
    pursuit_spawn_shared_radius: bool = False
    pursuit_body_frame_features: bool = True
    pursuit_neighbor_velocity_features: bool = True
    forward_sonar_enabled: bool = True
    forward_sonar_fov_degrees: float = 120.0
    forward_sonar_spacing_degrees: float = 15.0
    forward_sonar_range: float = 8.0
    evader_forward_sonar_enabled: bool = True
    evader_contact_distance: float = 1.0
    ri_dara_enabled: bool = True
    ri_dara_phase_inertia: float = 4.0
    ri_dara_role_check_interval: int = 1
    ri_dara_reassign_confirm_steps: int = 4
    ri_dara_reassign_hysteresis: float = 0.15
    ri_dara_reassign_cooldown_steps: int = 8
    ri_dara_angle_cost_weight: float = 1.0
    ri_dara_arrival_cost_weight: float = 0.35
    ri_dara_desired_radius_fraction: float = 0.75


@dataclass
class RewardConfig:
    """Current CEL-MADDPG/RI-DARA reward used by encirclement-v14."""
    version: str = "paper-task-v11"
    pursuit_weight: float = 0.6  # all-stage signed velocity projection toward the safe 3-5m band
    tracking_safe_distance: float = 3.0  # lower bound of the neutral tracking band
    tracking_balance_weight: float = 0.5  # mu3: group distance-spread penalty in tracking stage
    stall_penalty_rate: float = 0.4  # per second at zero speed outside the 5m capture radius
    stall_speed_fraction: float = 0.10  # penalty fades to zero at 10% of pursuer max speed
    tracking_distance_ratio: float = 10.0
    all_stage_distance_guidance: bool = True  # all-stage approach plus group distance cost in tracking stage
    enclosure_weight: float = 15.0
    enclosure_regression_weight: float = 5.0  # weaker penalty for temporary quality regressions
    tangential_weight: float = 0.25  # signed motion toward the larger neighboring angular gap
    enclosure_reward_mode: str = "progress"
    stage_isolated_rewards: bool = True
    enclosure_state_weight: float = 1.0  # positive high-quality bonus rate before dt scaling
    enclosure_bad_state_weight: float = 0.25  # reduced weight below the neutral quality threshold
    enclosure_positive_power: float = 2.0
    enclosure_quality_neutral: float = 0.7  # Phi below this is penalized; positive state reward starts above it
    enclosure_quality_goal: float = 0.9  # cap the positive state reward at this sufficient-quality goal
    contraction_weight: float = 1.5  # stronger inward progress after angular enclosure is established
    capture_reward: float = 300.0
    collision_penalty: float = 8.0
    # Retain the continuous hold signal for the 4-step safe capture.
    capture_progress_bonus: float = 12.0  # dense credit for each of the four consecutive safe-hold steps
    collision_failure_penalty: float = 0.0  # no extra collision failure penalty
    pursuer_contact_distance: float = 1.0  # center distance at/below which pursuers collide
    target_contact_distance: float = 0.0  # exact center overlap counts as target collision
    proximity_buffer_distance: float = 0.0  # disabled: unified sonar replaces inflated safety penalties
    proximity_penalty: float = 0.0
    sonar_avoidance_weight: float = 1.0  # one shared risk-change reward for all sonar hazards
    sonar_avoidance_clip: float = 0.2  # bound FOV-edge and nearest-hazard switches per step
    # Individual progress toward a persistent angular role.  This augments,
    # rather than replaces, the team enclosure objective.
    role_progress_weight: float = 2.0
    role_progress_clip: float = 0.2
    role_state_weight: float = 0.2
    desired_radius_fraction: float = 0.75
    angular_gate_floor: float = 0.1
    angular_gate_radius: float = 2.5


@dataclass
class PursuitTrainingConfig:
    max_steps: int = 480
    easy_episodes: int = 200
    medium_episodes: int = 400
    validation_interval: int = 50
    validation_episodes: int = 5
    checkpoint_interval: int = 25
    replay_capacity: int = 100_000
    curriculum: bool = False
    assignment_mode: str = "fixed"
    prioritized_replay: bool = True
    num_envs: int = 8  # eight parallel episodes per wave, one shared learner
    curriculum_mode: str = "success_gated"
    curriculum_min_episodes: int = 50
    curriculum_success_threshold: float = 0.6
    curriculum_success_streak: int = 3
    success_replay_fraction: float = 0.1
    success_trace_steps: int = 64
    update_interval: int = 10  # joint transitions, independent of worker count
    exploration_mode: str = "persistent_uniform"
    exploration_hold_steps_min: int = 3
    exploration_hold_steps_max: int = 5
    # RI-DARA already provides structured angular guidance. Large independent
    # perturbations mainly break the four-step hold and cooperative roles.
    exploration_noise_start: Tuple[float, float] = (0.40, 0.30)
    exploration_noise_min: Tuple[float, float] = (0.15, 0.10)
    exploration_decay_start_steps: int = 240_000  # keep full exploration for about 500 x 480-step episodes
    # After the 500-episode plateau, retain about 10% random actions at episode 5000.
    exploration_decay: float = 0.9999998122846898
    exploration_random_start: float = 0.15
    exploration_random_min: float = 0.05
    actor_warmup_updates: int = 200
    critic_hidden_dim: int = 128


@dataclass
class MATD3Config:
    hidden_dim: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.01
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 256
    replay_capacity: int = 500_000
    warmup_steps: int = 5_000
    explore_noise: float = 0.2
    prioritized_replay: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 200_000
    per_epsilon: float = 1e-5
    actor_activation: str = "tanh"  # pursuit selects softsign; old/target actors stay tanh
    actor_output_init: float = 0.0
    actor_hidden_activation: str = "relu"
    actor_lr_decay_steps: int = 0
    actor_lr_decay: float = 0.8
    actor_min_lr: float = 0.0
    critic_lr_decay_steps: int = 0
    critic_lr_decay: float = 0.33
    critic_min_lr: float = 0.0
    independent_critics: bool = False
    critic_hidden_dim: int = 0
    reward_scale: float = 1.0
    critic_loss: str = "mse"
    actor_warmup_updates: int = 0
    bootstrap_ramp_updates: int = 0
    critic_output_init: Optional[float] = None


@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    matd3: MATD3Config = field(default_factory=MATD3Config)
    pursuit: PursuitTrainingConfig = field(default_factory=PursuitTrainingConfig)
    seed: int = 7
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
