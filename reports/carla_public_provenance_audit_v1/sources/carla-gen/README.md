# CarlaGen - Generating Data with Carla 🚗

![img](assets/example.gif)

- [Configuration](#configuration)
  - [Local Configuration](#local-configuration)
- [Generating Data](#generating-data)
  - [Training Data](#training-data)
  - [Testing Data](#testing-data)
- [Generating Anomalous Scenarios](#generating-anomalous-scenarios)
  - [Adding Additional Semantic Classes](#adding-additional-semantic-classes)
  - [Flying Cars](#flying-cars)
  - [Crazy Traffic Lights](#crazy-traffic-lights)
  - [Combine](#combine)
  - [Overriding](#overriding)
- [Installation](#installation)
  - [Arch Linux](#arch-linux)
- [Carla Simulator Actor ID Synchronization](#carla-simulator-actor-id-synchronization)
- [Troubleshooting](#troubleshooting)


## Configuration 

Configuration is done with `hydra`. Config files are located in the `config` directory. 
There is no documentation at the moment, but most of these options should be self-explanatory. 
Make sure to read the documentation of hydra. 

Make sure that the config entry `simulation.carla.bin` points to your carla binary. Beware: the file you point this to is 
executed. 

When using the command line tool `sim.py`, you can use the command line to override config options.  
```shell
python sim.py simulation.pedestrians.number=10000 [...] 
```

Most of the work is done by hooking callbacks 
into the simulation. 

### Local Configuration 
You can add local configuration overrides to the 
`config/local/default.yaml` file. 
Such files will not be added to version control. 

For example, to set the location of your CARLA executable:
```yaml
# @package _global_

simulation:
  carla:
    bin: "/home/me/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    spawn: True
```

## Generating Data

### Training Data 
One-hour ride, captures data every couple of seconds. No motorbikes. 

```shell
python sim.py experiment=train
```

### Testing Data 
Shorter ride, captures every frame. Includes motorbikes.

```shell
python sim.py experiment=test
```

## Generating Anomalous Scenarios 
Anomalous behavior can be baked into the simulation via callbacks.

#### Flying Cars
```shell
python sim.py experiment=test callbacks=flyingcars
```

#### Crazy Traffic Lights
```shell
python sim.py experiment=test callbacks=trafficlight
```

#### Combine
```shell
python sim.py experiment=test callbacks=trafficlight,flyingcars
```

#### Overriding 

Find stuff [here](https://carla.readthedocs.io/en/latest/catalogue_props/).

```shell
python sim.py experiment=ano-spawn simulation.weather.sun_altitude_angle=0.0 callbacks.spawn_actor.filter="static.prop.advertisement"
```

## Installation 
There are a lot of dependencies.

First, download and set up the CARLA sim. Second, create an python environment, python 3.9 should work. 
Then, install dependencies:

```
pip install -r requirements.xml 
```

### Arch Linux
On arch, you will have to use a workaround to be able run carla.

```shell
pacman -S openmp
cd /usr/bin/
ln -s libomp.so libomp.so.5
```


## Patch: Carla Simulator Actor ID Synchronization

By default, the Carla simulator uses two different IDs for an actor: one for the simulator and another for the segmentation masks. This can cause inconsistencies when working with actor IDs. To resolve this issue, you need to modify the `ActorRegistry.cpp` file so that the IDs from the segmentation masks match those used by the simulator.

### Important Note

Attention: In the segmentation images, only 16 bits are used to encode the IDs. However, there are also IDs larger than \(2^{16} = 65536\). Therefore, in addition to the ID, the vehicle class should also be checked to ensure accurate identification.

### Steps to Modify `ActorRegistry.cpp`

1. **Locate the `ActorRegistry.cpp` file**:
   - The file is located at:
     ```
     <carla/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Actor/ActorRegistry.cpp>
     ```

2. **Edit the `ActorRegistry.cpp` file**:
   - Open the file in your preferred text editor.
   - Find the following line of code:
     ```cpp
     IdType Id = ++FActorRegistry::ID_COUNTER;
     ```
   - Comment out this line by adding `//` at the beginning:
     ```cpp
     // IdType Id = ++FActorRegistry::ID_COUNTER;
     ```
   - Replace the commented-out line with the following line:
     ```cpp
     IdType Id = Actor.GetUniqueID();
     ```

3. **Save and close the file**.

### Example

Your modified `ActorRegistry.cpp` should look like this:

```cpp
// IdType Id = ++FActorRegistry::ID_COUNTER;
IdType Id = Actor.GetUniqueID();
```


## Patch: Additional Classes (including Anomalies)

#### Adding Additional Semantic Classes

Instructions for adding additional semantic classes can be found [here](https://carla.readthedocs.io/en/latest/tuto_D_create_semantic_tags/).

#### Important

To ensure the classes match the anomaly defaults, pay attention to the class numbers. To use the defaults, modify the file: `<carla/LibCarla/source/carla/rpc/ObjectLabel.h>` so that the following class numbers are included:

```cpp
MyAnimals = 29u,
MyCars    = 30u,
```
#### Steps to Modify Class Numbers

1. Open the file `<carla/LibCarla/source/carla/rpc/ObjectLabel.h>` in your preferred text editor.
2. Add or modify the class numbers to match the following:

```cpp
MyAnimals = 29u,
MyCars    = 30u,
```
3. Follow the steps mentioned on the [website](https://carla.readthedocs.io/en/latest/tuto_D_create_semantic_tags/) for both MyAnimals and MyCars.

##### Steps to Manage Worlds with static Objects

1. **Move** `$TownXX` from `content/carla/maps` to your `$tmp` folder.
2. **Copy** the `TownXX.umap` file from `content/carla/maps_save/$townXX/my_cars` or `content/carla/maps_save/$townXX/my_animals` to `content/carla/maps`.
3. **Start** the simulation and load the world `$TownXX`.
4. **End** the simulation.
5. **Delete** `$TownXX` from `content/carla/maps`.
6. **Copy** `$TownXX` from the `$tmp` folder back to `content/carla/maps`.

By following these steps, you ensure that the static objects are correctly loaded into the simulation and that the original files are restored afterward.

## Troubleshooting 
After carla crashes, do 

```shell
killall CarlaUE4-Linux-Shipping
```

a couple of times. 


## Container 

```
# set gpu, can also be a uuid 
export CARLA_GPU_ID=1

# run containers in certain projects 
sudo -E docker compose -p carla-gpu1-1 up -d

# shell in client container  
sudo -E docker compose -p carla-gpu1-1 exec -it client bash

# generate some training data 
for seed in $(seq 100); do python sim.py experiment=train simulation.carla.spawn=False simulation.carla.host=carla simulation.world_name="Town02"; done 
```


NOTE: town01 has broken spawns 

