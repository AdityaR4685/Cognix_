#!/bin/sh
# Script used to generate the entire dataset.

#export CARLA_GPU_ID=0
## docker compose build --no-cache
#
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-1
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town01" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &
#
#sleep 60
#
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-2
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town02" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &
#
#sleep 60
#
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-3
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town03" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &
#
#sleep 60
#
#
#export CARLA_GPU_ID=2
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-1
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town04" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &
#
#sleep 60
#
## NOTE: Here, we additionally set simulation.pedestrians.pedestrians_cross_factor=1.0
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-2
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town05" seed=${seed} simulation.pedestrians.pedestrians_cross_factor=1.0; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &
#
#sleep 60
#
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-3
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town06" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &


export CARLA_GPU_ID=3
# docker compose build --no-cache

export PROJECT=carla-gpu${CARLA_GPU_ID}-train-4
cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town11" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

sleep 60

export PROJECT=carla-gpu${CARLA_GPU_ID}-train-5
cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town12" seed=${seed}; done'
sudo -E docker compose -p "$PROJECT" up -d --force-recreate
sleep 10
sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &

#sleep 60
#
#export PROJECT=carla-gpu${CARLA_GPU_ID}-train-3
#cmd='for seed in $(seq 1 50); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town10HD" seed=${seed}; done'
#sudo -E docker compose -p "$PROJECT" up -d --force-recreate
#sleep 10
#sudo -E docker compose -p "$PROJECT" exec -T client bash -lc "$cmd" >/dev/null 2>&1 &



