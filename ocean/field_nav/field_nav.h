#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FIELD_NAV_POLAR_ANGLE_BINS 32
#define FIELD_NAV_POLAR_DISTANCE_BINS 16
#define FIELD_NAV_POLAR_MAX_DISTANCE_M 24.0f
#define FIELD_NAV_CHANNELS 3
#define FIELD_NAV_VECTOR_FEATURES 7
#define FIELD_NAV_OBS_SIZE (FIELD_NAV_CHANNELS * FIELD_NAV_POLAR_ANGLE_BINS * FIELD_NAV_POLAR_DISTANCE_BINS + FIELD_NAV_VECTOR_FEATURES)
#define FIELD_NAV_MAX_OBJECTS 80
#define FIELD_NAV_PI 3.14159265358979323846f
#define FIELD_NAV_TWO_PI 6.28318530717958647692f

enum FieldNavKind {
    FIELD_NAV_TREE = 0,
    FIELD_NAV_BUSH = 1,
    FIELD_NAV_POTHOLE = 2,
    FIELD_NAV_PERSON = 3,
    FIELD_NAV_WALL = 4,
};

typedef struct Log {
    float score;
    float episode_return;
    float episode_length;
    float success_rate;
    float collision_rate;
    float curriculum_progress;
    float n;
} Log;

typedef struct FieldNavObject {
    int kind;
    float x;
    float y;
    float radius;
    float vx;
    float vy;
    float x2;
    float y2;
} FieldNavObject;

typedef struct FieldNav {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int max_steps;
    float map_extent_m;
    int polar_observation;
    int polar_angle_bins;
    int polar_distance_bins;
    float polar_max_distance_m;
    float world_size_m;
    float dt;
    float fixed_speed_mps;
    float max_turn_rate_rps;
    float min_goal_distance_m;
    float max_goal_distance_m;
    float goal_tolerance_m;
    float robot_radius_m;
    float inflation_radius_m;
    float near_obstacle_threshold_m;
    float reset_start_clearance_m;
    float reset_goal_clearance_m;
    float reset_forward_clearance_m;
    float reset_forward_margin_m;

    int tree_rows_min;
    int tree_rows_max;
    int bushes_min;
    int bushes_max;
    int potholes_min;
    int potholes_max;
    int people_min;
    int people_max;
    int walls_min;
    int walls_max;
    int curriculum_enabled;
    int curriculum_warmup_steps;
    int lifetime_steps;

    float robot_x;
    float robot_y;
    float heading;
    float speed;
    float yaw_rate;
    float prev_steering;
    float goal_x;
    float goal_y;
    float prev_goal_distance;
    int steps;

    FieldNavObject objects[FIELD_NAV_MAX_OBJECTS];
    int object_count;
    float episode_return;
    int episode_length;
} FieldNav;

static inline float fn_clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float fn_wrap_pi(float x) {
    while (x > FIELD_NAV_PI) x -= FIELD_NAV_TWO_PI;
    while (x < -FIELD_NAV_PI) x += FIELD_NAV_TWO_PI;
    return x;
}

static inline unsigned int fn_rand_u32(FieldNav* env) {
    unsigned int x = env->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng = x ? x : 2463534242u;
    return env->rng;
}

static inline float fn_rand01(FieldNav* env) {
    return (fn_rand_u32(env) >> 8) * (1.0f / 16777216.0f);
}

static inline float fn_uniform(FieldNav* env, float lo, float hi) {
    return lo + (hi - lo) * fn_rand01(env);
}

static inline int fn_randint(FieldNav* env, int lo, int hi_inclusive) {
    int span = hi_inclusive - lo + 1;
    return lo + (int)(fn_rand01(env) * (float)span);
}

static inline float fn_normal(FieldNav* env, float mean, float stddev) {
    float u1 = fmaxf(fn_rand01(env), 1e-6f);
    float u2 = fn_rand01(env);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(FIELD_NAV_TWO_PI * u2);
    return mean + stddev * z;
}

static inline float fn_distance(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return sqrtf(dx * dx + dy * dy);
}

static inline float fn_point_segment_distance(float px, float py, float x1, float y1, float x2, float y2) {
    float vx = x2 - x1;
    float vy = y2 - y1;
    float wx = px - x1;
    float wy = py - y1;
    float denom = vx * vx + vy * vy + 1e-8f;
    float t = fn_clip((wx * vx + wy * vy) / denom, 0.0f, 1.0f);
    float cx = x1 + t * vx;
    float cy = y1 + t * vy;
    return fn_distance(px, py, cx, cy);
}

static inline float fn_object_cost(int kind) {
    if (kind == FIELD_NAV_BUSH) return 0.42f;
    if (kind == FIELD_NAV_POTHOLE) return 0.65f;
    return 1.0f;
}

static inline int fn_observation_channel(int kind) {
    if (kind == FIELD_NAV_PERSON) return 1;
    if (kind == FIELD_NAV_BUSH || kind == FIELD_NAV_POTHOLE) return 2;
    return 0;
}

static inline int fn_clamp_int(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline void fn_polar_mark(FieldNav* env, float* obs, float local_x, float local_y, int kind, float radius) {
    float dist = sqrtf(local_x * local_x + local_y * local_y);
    if (dist > env->polar_max_distance_m) return;

    float angle = atan2f(local_y, local_x);
    int channel = fn_observation_channel(kind);
    float cost = fn_object_cost(kind);
    float radial_extent = fmaxf(radius, env->inflation_radius_m);
    float angular_extent = atan2f(radial_extent, fmaxf(dist, 1e-3f));

    int angle_bins = env->polar_angle_bins;
    int distance_bins = env->polar_distance_bins;
    int a_start = (int)floorf((angle - angular_extent + FIELD_NAV_PI) / FIELD_NAV_TWO_PI * (float)angle_bins);
    int a_end = (int)ceilf((angle + angular_extent + FIELD_NAV_PI) / FIELD_NAV_TWO_PI * (float)angle_bins);
    int d_start = (int)floorf((dist - radial_extent) / env->polar_max_distance_m * (float)distance_bins);
    int d_end = (int)ceilf((dist + radial_extent) / env->polar_max_distance_m * (float)distance_bins);

    if (d_end < 0 || d_start >= distance_bins) return;
    d_start = fn_clamp_int(d_start, 0, distance_bins - 1);
    d_end = fn_clamp_int(d_end, 0, distance_bins - 1);
    if (d_end < d_start) return;

    int a_len = a_end - a_start + 1;
    if (a_len <= 0) {
        a_len += angle_bins;
    }
    if (a_len >= angle_bins) {
        for (int ab = 0; ab < angle_bins; ab++) {
            int offset = channel * angle_bins * distance_bins + ab * distance_bins;
            for (int db = d_start; db <= d_end; db++) {
                int idx = offset + db;
                if (cost > obs[idx]) obs[idx] = cost;
            }
        }
        return;
    }

    for (int i = 0; i < a_len; i++) {
        int ab = a_start + i;
        while (ab < 0) ab += angle_bins;
        while (ab >= angle_bins) ab -= angle_bins;
        int offset = channel * angle_bins * distance_bins + ab * distance_bins;
        for (int db = d_start; db <= d_end; db++) {
            int idx = offset + db;
            if (cost > obs[idx]) obs[idx] = cost;
        }
    }
}

static inline void fn_polar_mark_circle(FieldNav* env, float* obs, const FieldNavObject* obj) {
    int sample_count = obj->kind == FIELD_NAV_TREE ? 6 : 4;
    for (int i = 0; i < sample_count; i++) {
        float theta = FIELD_NAV_TWO_PI * (float)i / (float)sample_count;
        float px = obj->x + cosf(theta) * obj->radius;
        float py = obj->y + sinf(theta) * obj->radius;
        float dx = px - env->robot_x;
        float dy = py - env->robot_y;
        float local_x = dx * cosf(env->heading) + dy * sinf(env->heading);
        float local_y = -dx * sinf(env->heading) + dy * cosf(env->heading);
        fn_polar_mark(env, obs, local_x, local_y, obj->kind, obj->radius);
    }
    fn_polar_mark(env, obs, obj->x - env->robot_x, obj->y - env->robot_y, obj->kind, obj->radius);
}

static inline void fn_polar_mark_wall(FieldNav* env, float* obs, const FieldNavObject* obj) {
    float seg_len = fn_distance(obj->x, obj->y, obj->x2, obj->y2);
    int sample_count = (int)ceilf(seg_len / 1.5f);
    if (sample_count < 3) sample_count = 3;
    for (int i = 0; i <= sample_count; i++) {
        float t = (float)i / (float)sample_count;
        float px = obj->x + t * (obj->x2 - obj->x);
        float py = obj->y + t * (obj->y2 - obj->y);
        float dx = px - env->robot_x;
        float dy = py - env->robot_y;
        float local_x = dx * cosf(env->heading) + dy * sinf(env->heading);
        float local_y = -dx * sinf(env->heading) + dy * cosf(env->heading);
        fn_polar_mark(env, obs, local_x, local_y, obj->kind, 0.5f * obj->radius);
    }
}

static inline int fn_is_collidable(int kind) {
    return kind == FIELD_NAV_TREE || kind == FIELD_NAV_PERSON || kind == FIELD_NAV_WALL;
}

static inline float fn_object_margin(float x, float y, const FieldNavObject* obj) {
    if (obj->kind == FIELD_NAV_WALL) {
        return fn_point_segment_distance(x, y, obj->x, obj->y, obj->x2, obj->y2) - 0.5f * obj->radius;
    }
    return fn_distance(x, y, obj->x, obj->y) - obj->radius;
}

static inline float fn_point_collidable_clearance(FieldNav* env, float x, float y) {
    float min_clearance = 1e9f;
    for (int i = 0; i < env->object_count; i++) {
        FieldNavObject* obj = &env->objects[i];
        if (!fn_is_collidable(obj->kind)) continue;
        float clearance = fn_object_margin(x, y, obj) - env->robot_radius_m;
        if (clearance < min_clearance) {
            min_clearance = clearance;
        }
    }
    return min_clearance;
}

static inline int fn_forward_path_clear(FieldNav* env) {
    if (env->reset_forward_clearance_m <= 0.0f) return 1;

    float step_distance = env->fixed_speed_mps * env->dt;
    int sample_count = (int)ceilf(env->reset_forward_clearance_m / fmaxf(step_distance, 1e-6f));
    if (sample_count < 2) sample_count = 2;
    float cos_h = cosf(env->heading);
    float sin_h = sinf(env->heading);

    for (int i = 1; i <= sample_count; i++) {
        float distance = env->reset_forward_clearance_m * (float)i / (float)sample_count;
        float x = env->robot_x + cos_h * distance;
        float y = env->robot_y + sin_h * distance;
        if (fn_point_collidable_clearance(env, x, y) < env->reset_forward_margin_m) {
            return 0;
        }
    }
    return 1;
}

static inline int fn_reset_layout_valid(FieldNav* env) {
    if (
        env->reset_start_clearance_m > 0.0f &&
        fn_point_collidable_clearance(env, env->robot_x, env->robot_y) < env->reset_start_clearance_m
    ) {
        return 0;
    }
    if (
        env->reset_goal_clearance_m > 0.0f &&
        fn_point_collidable_clearance(env, env->goal_x, env->goal_y) < env->reset_goal_clearance_m
    ) {
        return 0;
    }
    return fn_forward_path_clear(env);
}

static inline float fn_curriculum_progress(FieldNav* env) {
    if (!env->curriculum_enabled || env->curriculum_warmup_steps <= 0) {
        return 1.0f;
    }
    return fn_clip((float)env->lifetime_steps / (float)env->curriculum_warmup_steps, 0.0f, 1.0f);
}

static inline int fn_curriculum_int(float progress, int start, int end) {
    float value = (float)start + ((float)end - (float)start) * progress;
    return (int)floorf(value + 0.5f);
}

static inline float fn_curriculum_float(float progress, float start, float end) {
    return start + (end - start) * progress;
}

static inline void fn_add_circle(FieldNav* env, int kind, float x, float y, float radius, float vx, float vy) {
    if (env->object_count >= FIELD_NAV_MAX_OBJECTS) return;
    FieldNavObject* obj = &env->objects[env->object_count++];
    obj->kind = kind;
    obj->x = x;
    obj->y = y;
    obj->radius = radius;
    obj->vx = vx;
    obj->vy = vy;
    obj->x2 = 0.0f;
    obj->y2 = 0.0f;
}

static inline void fn_add_wall(FieldNav* env, float x1, float y1, float x2, float y2, float thickness) {
    if (env->object_count >= FIELD_NAV_MAX_OBJECTS) return;
    FieldNavObject* obj = &env->objects[env->object_count++];
    obj->kind = FIELD_NAV_WALL;
    obj->x = x1;
    obj->y = y1;
    obj->x2 = x2;
    obj->y2 = y2;
    obj->radius = thickness;
    obj->vx = 0.0f;
    obj->vy = 0.0f;
}

static inline void fn_sample_objects(FieldNav* env) {
    float span = env->world_size_m * 0.42f;
    env->object_count = 0;
    float progress = fn_curriculum_progress(env);
    int tree_rows_min = fn_curriculum_int(progress, 1, env->tree_rows_min);
    int tree_rows_max = fn_curriculum_int(progress, 2, env->tree_rows_max);
    int bushes_min = fn_curriculum_int(progress, 3, env->bushes_min);
    int bushes_max = fn_curriculum_int(progress, 8, env->bushes_max);
    int potholes_min = fn_curriculum_int(progress, 1, env->potholes_min);
    int potholes_max = fn_curriculum_int(progress, 4, env->potholes_max);
    int people_min = fn_curriculum_int(progress, 0, env->people_min);
    int people_max = fn_curriculum_int(progress, 2, env->people_max);
    int walls_min = fn_curriculum_int(progress, 0, env->walls_min);
    int walls_max = fn_curriculum_int(progress, 1, env->walls_max);
    if (tree_rows_max < tree_rows_min) tree_rows_max = tree_rows_min;
    if (bushes_max < bushes_min) bushes_max = bushes_min;
    if (potholes_max < potholes_min) potholes_max = potholes_min;
    if (people_max < people_min) people_max = people_min;
    if (walls_max < walls_min) walls_max = walls_min;

    int rows = fn_randint(env, tree_rows_min, tree_rows_max);
    for (int row = 0; row < rows; row++) {
        float y = fn_uniform(env, -span, span);
        float x0 = fn_uniform(env, -span, -span * 0.2f);
        float spacing = fn_uniform(env, 2.2f, 3.6f);
        int count = fn_randint(env, 4, 8);
        float jitter = fn_uniform(env, 0.0f, 0.35f);
        for (int i = 0; i < count; i++) {
            float x = x0 + i * spacing + fn_normal(env, 0.0f, jitter);
            if (fabsf(x) > span) continue;
            fn_add_circle(env, FIELD_NAV_TREE, x, y + fn_normal(env, 0.0f, jitter), fn_uniform(env, 0.35f, 0.75f), 0.0f, 0.0f);
        }
    }

    int bushes = fn_randint(env, bushes_min, bushes_max);
    for (int i = 0; i < bushes; i++) {
        fn_add_circle(env, FIELD_NAV_BUSH, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.45f, 1.25f), 0.0f, 0.0f);
    }

    int potholes = fn_randint(env, potholes_min, potholes_max);
    for (int i = 0; i < potholes; i++) {
        fn_add_circle(env, FIELD_NAV_POTHOLE, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.35f, 0.95f), 0.0f, 0.0f);
    }

    int people = fn_randint(env, people_min, people_max);
    for (int i = 0; i < people; i++) {
        float heading = fn_uniform(env, -FIELD_NAV_PI, FIELD_NAV_PI);
        float speed = fn_uniform(env, 0.15f, 0.55f);
        fn_add_circle(env, FIELD_NAV_PERSON, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.32f, 0.45f), speed * cosf(heading), speed * sinf(heading));
    }

    int walls = fn_randint(env, walls_min, walls_max);
    for (int i = 0; i < walls; i++) {
        float cx = fn_uniform(env, -span, span);
        float cy = fn_uniform(env, -span, span);
        float length = fn_uniform(env, 4.0f, 10.0f);
        float angle = fn_uniform(env, -FIELD_NAV_PI, FIELD_NAV_PI);
        float dx = 0.5f * length * cosf(angle);
        float dy = 0.5f * length * sinf(angle);
        fn_add_wall(env, cx - dx, cy - dy, cx + dx, cy + dy, fn_uniform(env, 0.25f, 0.55f));
    }
}

static inline float fn_goal_distance(FieldNav* env) {
    return fn_distance(env->robot_x, env->robot_y, env->goal_x, env->goal_y);
}

static inline void fn_rasterize_costmap(FieldNav* env, float* obs, int map_size, float map_extent_m) {
    int map_cells = map_size * map_size;
    float half = map_extent_m * 0.5f;
    float step = map_extent_m / (float)(map_size - 1);
    float cos_h = cosf(env->heading);
    float sin_h = sinf(env->heading);

    for (int i = 0; i < env->object_count; i++) {
        FieldNavObject* obj = &env->objects[i];
        float expansion = env->inflation_radius_m + obj->radius;
        float min_x = obj->x - expansion;
        float max_x = obj->x + expansion;
        float min_y = obj->y - expansion;
        float max_y = obj->y + expansion;

        if (obj->kind == FIELD_NAV_WALL) {
            expansion = env->inflation_radius_m + 0.5f * obj->radius;
            min_x = fminf(obj->x, obj->x2) - expansion;
            max_x = fmaxf(obj->x, obj->x2) + expansion;
            min_y = fminf(obj->y, obj->y2) - expansion;
            max_y = fmaxf(obj->y, obj->y2) + expansion;
        }

        float local_min_x = 1e9f;
        float local_max_x = -1e9f;
        float local_min_y = 1e9f;
        float local_max_y = -1e9f;
        float corners_x[4] = {min_x, min_x, max_x, max_x};
        float corners_y[4] = {min_y, max_y, min_y, max_y};
        for (int c = 0; c < 4; c++) {
            float dx = corners_x[c] - env->robot_x;
            float dy = corners_y[c] - env->robot_y;
            float lx = dx * cos_h + dy * sin_h;
            float ly = -dx * sin_h + dy * cos_h;
            if (lx < local_min_x) local_min_x = lx;
            if (lx > local_max_x) local_max_x = lx;
            if (ly < local_min_y) local_min_y = ly;
            if (ly > local_max_y) local_max_y = ly;
        }
        if (local_max_x < -half || local_min_x > half || local_max_y < -half || local_min_y > half) {
            continue;
        }

        int x0 = fn_clamp_int((int)floorf((local_min_x + half) / step), 0, map_size - 1);
        int x1 = fn_clamp_int((int)ceilf((local_max_x + half) / step), 0, map_size - 1);
        int y0 = fn_clamp_int((int)floorf((local_min_y + half) / step), 0, map_size - 1);
        int y1 = fn_clamp_int((int)ceilf((local_max_y + half) / step), 0, map_size - 1);
        if (x1 < x0 || y1 < y0) continue;

        for (int oy = y0; oy <= y1; oy++) {
            float gy = -half + step * (float)oy;
            for (int ox = x0; ox <= x1; ox++) {
                float gx = -half + step * (float)ox;
                float wx = env->robot_x + gx * cos_h - gy * sin_h;
                float wy = env->robot_y + gx * sin_h + gy * cos_h;
                float margin = fn_object_margin(wx, wy, obj);
                if (margin >= env->inflation_radius_m) continue;
                float cost = margin <= 0.0f ? fn_object_cost(obj->kind) : fn_object_cost(obj->kind) * (1.0f - margin / env->inflation_radius_m);
                cost = fn_clip(cost, 0.0f, 1.0f);
                int cell = oy * map_size + ox;
                int offset = fn_observation_channel(obj->kind) * map_cells + cell;
                if (cost > obs[offset]) obs[offset] = cost;
            }
        }
    }
}

static inline void fn_compute_observation(FieldNav* env) {
    float* obs = env->observations;
    for (int i = 0; i < FIELD_NAV_OBS_SIZE; i++) {
        obs[i] = 0.0f;
    }

    int map_cells = env->polar_angle_bins * env->polar_distance_bins;
    for (int i = 0; i < env->object_count; i++) {
        FieldNavObject* obj = &env->objects[i];
        if (obj->kind == FIELD_NAV_WALL) {
            fn_polar_mark_wall(env, obs, obj);
        } else {
            fn_polar_mark_circle(env, obs, obj);
        }
    }

    int base = FIELD_NAV_CHANNELS * map_cells;
    float dx = env->goal_x - env->robot_x;
    float dy = env->goal_y - env->robot_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float goal_heading = atan2f(dy, dx);
    float heading_error = fn_wrap_pi(goal_heading - env->heading);
    obs[base + 0] = fn_clip(dist / 25.0f, 0.0f, 1.0f);
    obs[base + 1] = sinf(heading_error);
    obs[base + 2] = cosf(heading_error);
    obs[base + 3] = fn_clip(env->speed / 3.0f, -1.0f, 1.0f);
    obs[base + 4] = fn_clip(env->yaw_rate / env->max_turn_rate_rps, -1.0f, 1.0f);
    obs[base + 5] = fn_clip(env->prev_steering, -1.0f, 1.0f);
    obs[base + 6] = 1.0f;
}

static inline void c_reset(FieldNav* env) {
    env->steps = 0;
    env->prev_steering = 0.0f;
    env->speed = env->fixed_speed_mps;
    env->yaw_rate = 0.0f;
    env->episode_return = 0.0f;
    env->episode_length = 0;

    env->robot_x = fn_uniform(env, -3.0f, 3.0f);
    env->robot_y = fn_uniform(env, -3.0f, 3.0f);
    env->heading = fn_uniform(env, -FIELD_NAV_PI, FIELD_NAV_PI);
    float curriculum_progress = fn_curriculum_progress(env);
    float min_goal_distance = fn_curriculum_float(curriculum_progress, 6.0f, env->min_goal_distance_m);
    float max_goal_distance = fn_curriculum_float(curriculum_progress, 12.0f, env->max_goal_distance_m);
    if (max_goal_distance < min_goal_distance) max_goal_distance = min_goal_distance;

    int found = 0;
    for (int i = 0; i < 200; i++) {
        float gx = fn_uniform(env, -env->world_size_m * 0.4f, env->world_size_m * 0.4f);
        float gy = fn_uniform(env, -env->world_size_m * 0.4f, env->world_size_m * 0.4f);
        float d = fn_distance(env->robot_x, env->robot_y, gx, gy);
        if (d >= min_goal_distance && d <= max_goal_distance) {
            env->goal_x = gx;
            env->goal_y = gy;
            found = 1;
            break;
        }
    }
    if (!found) {
        env->goal_x = env->robot_x + max_goal_distance;
        env->goal_y = env->robot_y;
    }

    for (int i = 0; i < 64; i++) {
        fn_sample_objects(env);
        if (fn_reset_layout_valid(env)) break;
    }
    env->prev_goal_distance = fn_goal_distance(env);
    fn_compute_observation(env);
}

static inline void fn_add_episode_log(FieldNav* env, int success, int collision) {
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.success_rate += (float)success;
    env->log.collision_rate += (float)collision;
    env->log.curriculum_progress += fn_curriculum_progress(env);
    env->log.n += 1.0f;
}

static inline void c_step(FieldNav* env) {
    static const float steering_values[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int action = (int)env->actions[0];
    if (action < 0) action = 0;
    if (action > 4) action = 4;
    float steering = steering_values[action];

    env->yaw_rate = steering * env->max_turn_rate_rps;
    env->heading = fn_wrap_pi(env->heading + env->yaw_rate * env->dt);
    env->speed = env->fixed_speed_mps;
    env->robot_x += cosf(env->heading) * env->speed * env->dt;
    env->robot_y += sinf(env->heading) * env->speed * env->dt;
    env->steps += 1;
    env->lifetime_steps += 1;
    env->episode_length += 1;

    float half_world = env->world_size_m * 0.5f;
    for (int i = 0; i < env->object_count; i++) {
        FieldNavObject* obj = &env->objects[i];
        if (obj->kind != FIELD_NAV_PERSON) continue;
        obj->x += obj->vx * env->dt;
        obj->y += obj->vy * env->dt;
        if (obj->x < -half_world || obj->x > half_world) {
            obj->x = fn_clip(obj->x, -half_world, half_world);
            obj->vx *= -1.0f;
        }
        if (obj->y < -half_world || obj->y > half_world) {
            obj->y = fn_clip(obj->y, -half_world, half_world);
            obj->vy *= -1.0f;
        }
    }

    float goal_distance = fn_goal_distance(env);
    float progress = env->prev_goal_distance - goal_distance;
    env->prev_goal_distance = goal_distance;

    float nearest_margin = 1e9f;
    float semantic_penalty = 0.0f;
    for (int i = 0; i < env->object_count; i++) {
        FieldNavObject* obj = &env->objects[i];
        float margin = fn_object_margin(env->robot_x, env->robot_y, obj);
        if (fn_is_collidable(obj->kind) && margin < nearest_margin) {
            nearest_margin = margin;
        }
        float robot_margin = margin - env->robot_radius_m;
        if (obj->kind == FIELD_NAV_BUSH && robot_margin < 0.25f) {
            semantic_penalty += 0.08f * (0.25f - robot_margin);
        } else if (obj->kind == FIELD_NAV_POTHOLE && robot_margin < 0.0f) {
            semantic_penalty += 1.25f * (-robot_margin);
        } else if (obj->kind == FIELD_NAV_PERSON && robot_margin < 1.2f) {
            semantic_penalty += 0.35f * (1.2f - robot_margin);
        }
    }
    nearest_margin -= env->robot_radius_m;

    int collision = nearest_margin <= 0.0f;
    int goal_reached = goal_distance <= env->goal_tolerance_m;
    int out_of_bounds = fabsf(env->robot_x) > half_world || fabsf(env->robot_y) > half_world;

    float reward = progress;
    if (goal_reached) reward += 20.0f;
    if (collision) reward -= 20.0f;
    if (nearest_margin < env->near_obstacle_threshold_m) {
        reward -= 0.2f * (env->near_obstacle_threshold_m - nearest_margin);
    }
    reward -= 0.02f * fabsf(steering - env->prev_steering);
    reward -= 0.01f;
    reward -= semantic_penalty;
    if (out_of_bounds) reward -= 10.0f;
    env->prev_steering = steering;

    int done = collision || goal_reached || out_of_bounds;
    int truncated = env->steps >= env->max_steps;
    env->rewards[0] = reward;
    env->terminals[0] = (float)(done || truncated);
    env->episode_return += reward;

    if (done || truncated) {
        fn_add_episode_log(env, goal_reached, collision);
        c_reset(env);
    } else {
        fn_compute_observation(env);
    }
}

static inline void c_render(FieldNav* env) {
    printf(
        "field_nav step=%d pos=(%.2f,%.2f) heading=%.2f goal_dist=%.2f objects=%d\n",
        env->steps,
        env->robot_x,
        env->robot_y,
        env->heading,
        fn_goal_distance(env),
        env->object_count
    );
}

static inline void c_close(FieldNav* env) {
    (void)env;
}
