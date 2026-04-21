#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FIELD_NAV_MAP_SIZE 64
#define FIELD_NAV_CHANNELS 5
#define FIELD_NAV_VECTOR_FEATURES 7
#define FIELD_NAV_OBS_SIZE (FIELD_NAV_CHANNELS * FIELD_NAV_MAP_SIZE * FIELD_NAV_MAP_SIZE + FIELD_NAV_VECTOR_FEATURES)
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

static inline int fn_is_collidable(int kind) {
    return kind == FIELD_NAV_TREE || kind == FIELD_NAV_PERSON || kind == FIELD_NAV_WALL;
}

static inline float fn_object_margin(float x, float y, const FieldNavObject* obj) {
    if (obj->kind == FIELD_NAV_WALL) {
        return fn_point_segment_distance(x, y, obj->x, obj->y, obj->x2, obj->y2) - 0.5f * obj->radius;
    }
    return fn_distance(x, y, obj->x, obj->y) - obj->radius;
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

    int rows = fn_randint(env, env->tree_rows_min, env->tree_rows_max);
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

    int bushes = fn_randint(env, env->bushes_min, env->bushes_max);
    for (int i = 0; i < bushes; i++) {
        fn_add_circle(env, FIELD_NAV_BUSH, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.45f, 1.25f), 0.0f, 0.0f);
    }

    int potholes = fn_randint(env, env->potholes_min, env->potholes_max);
    for (int i = 0; i < potholes; i++) {
        fn_add_circle(env, FIELD_NAV_POTHOLE, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.35f, 0.95f), 0.0f, 0.0f);
    }

    int people = fn_randint(env, env->people_min, env->people_max);
    for (int i = 0; i < people; i++) {
        float heading = fn_uniform(env, -FIELD_NAV_PI, FIELD_NAV_PI);
        float speed = fn_uniform(env, 0.15f, 0.55f);
        fn_add_circle(env, FIELD_NAV_PERSON, fn_uniform(env, -span, span), fn_uniform(env, -span, span), fn_uniform(env, 0.32f, 0.45f), speed * cosf(heading), speed * sinf(heading));
    }

    int walls = fn_randint(env, env->walls_min, env->walls_max);
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

static inline void fn_compute_observation(FieldNav* env) {
    float* obs = env->observations;
    int map_cells = FIELD_NAV_MAP_SIZE * FIELD_NAV_MAP_SIZE;
    for (int i = 0; i < FIELD_NAV_CHANNELS * map_cells + FIELD_NAV_VECTOR_FEATURES; i++) {
        obs[i] = 0.0f;
    }

    float half = env->map_extent_m * 0.5f;
    float step = env->map_extent_m / (float)(FIELD_NAV_MAP_SIZE - 1);
    float cos_h = cosf(env->heading);
    float sin_h = sinf(env->heading);

    for (int oy = 0; oy < FIELD_NAV_MAP_SIZE; oy++) {
        float gy = -half + step * (float)oy;
        for (int ox = 0; ox < FIELD_NAV_MAP_SIZE; ox++) {
            float gx = -half + step * (float)ox;
            float wx = env->robot_x + gx * cos_h - gy * sin_h;
            float wy = env->robot_y + gx * sin_h + gy * cos_h;
            int cell = oy * FIELD_NAV_MAP_SIZE + ox;

            for (int i = 0; i < env->object_count; i++) {
                FieldNavObject* obj = &env->objects[i];
                float margin = fn_object_margin(wx, wy, obj);
                if (margin >= env->inflation_radius_m) continue;
                float cost = margin <= 0.0f ? fn_object_cost(obj->kind) : fn_object_cost(obj->kind) * (1.0f - margin / env->inflation_radius_m);
                cost = fn_clip(cost, 0.0f, 1.0f);
                int offset = obj->kind * map_cells + cell;
                if (cost > obs[offset]) obs[offset] = cost;
            }
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

    int found = 0;
    for (int i = 0; i < 200; i++) {
        float gx = fn_uniform(env, -env->world_size_m * 0.4f, env->world_size_m * 0.4f);
        float gy = fn_uniform(env, -env->world_size_m * 0.4f, env->world_size_m * 0.4f);
        float d = fn_distance(env->robot_x, env->robot_y, gx, gy);
        if (d >= env->min_goal_distance_m && d <= env->max_goal_distance_m) {
            env->goal_x = gx;
            env->goal_y = gy;
            found = 1;
            break;
        }
    }
    if (!found) {
        env->goal_x = env->robot_x + env->max_goal_distance_m;
        env->goal_y = env->robot_y;
    }

    fn_sample_objects(env);
    env->prev_goal_distance = fn_goal_distance(env);
    fn_compute_observation(env);
}

static inline void fn_add_episode_log(FieldNav* env, int success, int collision) {
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.success_rate += (float)success;
    env->log.collision_rate += (float)collision;
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
