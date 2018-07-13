# starcraft_agents

Starcraft II agents - coded using DeepMind PYSC2

## Installation

```
cd starcraft_agents
pip install -e .
```

## Application

Run Agent:

```
python -m pysc2.bin.agent --map Simple64 --agent random_agents.RandomAgent001 --agent_race terran
```

Train Agent:

```
python -m starcraft_agents.bin.dqn_agent --map MoveToBeacon --agent move_to_beacon_agents.MoveToBeaconAgent007 --agent_race terran
```
