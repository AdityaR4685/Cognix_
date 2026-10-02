#!/bin/sh
# Script used to generate the entire dataset.

export CARLA_GPU_ID=1
# docker compose build --no-cache

export PROJECT=carla-gpu1-1
cmd='for seed in $(seq 150 200); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10

sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu1-2
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-02-vanish-actor simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu1-3
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-03-instant-weather simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60


export CARLA_GPU_ID=2
export PROJECT=carla-gpu2-1
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-04-spawn-props simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

# NOTE: Here, we additionally set simulation.pedestrians.pedestrians_cross_factor=1.0
export PROJECT=carla-gpu2-2
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-05-running-pedestrians simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed} simulation.pedestrians.pedestrians_cross_factor=1.0; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu2-3
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-06-crazy-cars simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &


export CARLA_GPU_ID=3
# docker compose build --no-cache

export PROJECT=carla-gpu3-1
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-07-streetlight-flicker simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu3-2
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-08-trafficlight-flicker simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu3-3
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-09-trafficlight-blink-yellow simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export CARLA_GPU_ID=4

export PROJECT=carla-gpu4-1
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-10-trafficlight-off simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu4-2
cmd='for seed in $(seq 150 200); do python sim.py experiment=ano-test-11-driver-steering simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

