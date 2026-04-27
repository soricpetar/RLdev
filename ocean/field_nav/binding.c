#include "field_nav.h"

#define OBS_SIZE FIELD_NAV_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T FloatTensor

#define Env FieldNav
#include "vecenv.h"

static int env_get_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (int)item->value;
}

static float env_get_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item == NULL ? fallback : (float)item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;

    env->max_steps = env_get_int(kwargs, "max_steps", 400);
    env->map_extent_m = env_get_float(kwargs, "map_extent_m", 12.0f);
    env->polar_observation = env_get_int(kwargs, "polar_observation", 1);
    env->polar_angle_bins = env_get_int(kwargs, "polar_angle_bins", FIELD_NAV_POLAR_ANGLE_BINS);
    env->polar_distance_bins = env_get_int(kwargs, "polar_distance_bins", FIELD_NAV_POLAR_DISTANCE_BINS);
    env->polar_max_distance_m = env_get_float(kwargs, "polar_max_distance_m", FIELD_NAV_POLAR_MAX_DISTANCE_M);
    env->world_size_m = env_get_float(kwargs, "world_size_m", 50.0f);
    env->dt = env_get_float(kwargs, "dt", 0.2f);
    env->fixed_speed_mps = env_get_float(kwargs, "fixed_speed_mps", 1.0f);
    env->max_turn_rate_rps = env_get_float(kwargs, "max_turn_rate_rps", 1.0f);
    env->min_goal_distance_m = env_get_float(kwargs, "min_goal_distance_m", 10.0f);
    env->max_goal_distance_m = env_get_float(kwargs, "max_goal_distance_m", 22.0f);
    env->goal_tolerance_m = env_get_float(kwargs, "goal_tolerance_m", 0.65f);
    env->robot_radius_m = env_get_float(kwargs, "robot_radius_m", 0.35f);
    env->inflation_radius_m = env_get_float(kwargs, "inflation_radius_m", 1.0f);
    env->near_obstacle_threshold_m = env_get_float(kwargs, "near_obstacle_threshold_m", 2.0f);
    env->reset_start_clearance_m = env_get_float(kwargs, "reset_start_clearance_m", 0.0f);
    env->reset_goal_clearance_m = env_get_float(kwargs, "reset_goal_clearance_m", 0.0f);
    env->reset_forward_clearance_m = env_get_float(kwargs, "reset_forward_clearance_m", 0.0f);
    env->reset_forward_margin_m = env_get_float(kwargs, "reset_forward_margin_m", 0.0f);

    env->tree_rows_min = env_get_int(kwargs, "tree_rows_min", 2);
    env->tree_rows_max = env_get_int(kwargs, "tree_rows_max", 4);
    env->bushes_min = env_get_int(kwargs, "bushes_min", 6);
    env->bushes_max = env_get_int(kwargs, "bushes_max", 14);
    env->potholes_min = env_get_int(kwargs, "potholes_min", 4);
    env->potholes_max = env_get_int(kwargs, "potholes_max", 10);
    env->people_min = env_get_int(kwargs, "people_min", 2);
    env->people_max = env_get_int(kwargs, "people_max", 6);
    env->walls_min = env_get_int(kwargs, "walls_min", 1);
    env->walls_max = env_get_int(kwargs, "walls_max", 4);
    env->curriculum_enabled = env_get_int(kwargs, "curriculum_enabled", 0);
    env->curriculum_warmup_steps = env_get_int(kwargs, "curriculum_warmup_steps", 0);
    env->lifetime_steps = 0;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "curriculum_progress", log->curriculum_progress);
}
