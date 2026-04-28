#include "field_nav.h"

int main(void) {
    FieldNav env = {0};
    float observations[FIELD_NAV_OBS_SIZE] = {0};
    float actions[1] = {2};
    float rewards[1] = {0};
    float terminals[1] = {0};

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.rng = 1;
    env.max_steps = 400;
    env.map_extent_m = 12.0f;
    env.polar_observation = 1;
    env.polar_angle_bins = FIELD_NAV_POLAR_ANGLE_BINS;
    env.polar_distance_bins = FIELD_NAV_POLAR_DISTANCE_BINS;
    env.polar_max_distance_m = FIELD_NAV_POLAR_MAX_DISTANCE_M;
    env.world_size_m = 50.0f;
    env.dt = 0.2f;
    env.fixed_speed_mps = 1.0f;
    env.min_speed_mps = 0.0f;
    env.max_speed_mps = 1.6f;
    env.acceleration_mps2 = 1.5f;
    env.brake_deceleration_mps2 = 2.5f;
    env.coast_deceleration_mps2 = 0.3f;
    env.max_turn_rate_rps = 1.0f;
    env.min_goal_distance_m = 10.0f;
    env.max_goal_distance_m = 22.0f;
    env.goal_tolerance_m = 0.65f;
    env.robot_radius_m = 0.35f;
    env.inflation_radius_m = 1.0f;
    env.near_obstacle_threshold_m = 2.0f;
    env.reset_start_clearance_m = 1.0f;
    env.reset_goal_clearance_m = 1.0f;
    env.reset_forward_clearance_m = 1.5f;
    env.reset_forward_margin_m = 0.25f;
    env.tree_rows_min = 2;
    env.tree_rows_max = 4;
    env.bushes_min = 6;
    env.bushes_max = 14;
    env.potholes_min = 4;
    env.potholes_max = 10;
    env.people_min = 2;
    env.people_max = 6;
    env.walls_min = 1;
    env.walls_max = 4;

    c_reset(&env);
    for (int i = 0; i < 32; i++) {
        env.actions[0] = (float)(i % 15);
        c_step(&env);
        if (env.terminals[0] > 0.5f) {
            printf("done reward=%.3f\n", env.rewards[0]);
        }
    }
    c_render(&env);
    return 0;
}
