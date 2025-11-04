# Configuring Rollouts with Hydra and Pydantic

The present tutorial touches on three aspects required for robust and maintainable RL development:
- **Hydra's composition system** for modular configurations
- **Pydantic's validation magic** for type-safe data models
- **Configured rollout scripts** used in RL inference pipelines

---

## 📚 1.The Configuration Problem

Imagine you're running RL experiments. You need to configure:
- Environment parameters (reward types, episode lengths)
- Training hyperparameters 
- Rollout settings
- Logging preferences

Instead of hard-coded values scattered throughout your code, a more robust and maintable approach 
consists in using __centralized__, __validated__, __composable configurations ✨

> **💡 Info Point - Configuration Management**: The practice of systematically handling settings and parameters in software systems. Critical for reproducible research and experiment tracking. <br>

**📢 Action Points**:  
+ Check out the [hydra official documentation](https://hydra.cc/). 
+ Check out the [pydantic official documentation](https://docs.pydantic.dev/latest/).

---

## 📚 2. Hydra - The Composition Powerhouse

Take a look at `configs/config.yaml`:

```4:8:configs/config.yaml
defaults: 
  - _self_
```



> **💡 Info Point - Hydra Defaults**: The __defaults__ is a special reserved key field that defines which configuration files or groups should be 
composed together to form the final configuration at runtime. Here, __\_self\___ means that the content of 
the current files is actually part of the config.

> **💡 Info Point - Hydra Defaults**: To extend the list and add composition, you need to first 
create subfolders in the configs directory, populate them with config files and then add pointers to those 
files in the defaults field.

**📢 Action Points**:  

+ Create an environment configuration `env_config.yaml` under the folder `environment`. Populate it with the following: 

```
max_episode_steps: 50
render_mode: # Can be human, rgb_array or null
reward_type: dense # Can be dense or sparse
```
+ Create a rollout configuration `rollout_config.yaml` under the folder `rollout`. Populate it with the following: 

```
rollout_unit: episodes # Can either be episodes, steps or time (in seconds)
rollout_unit_value: 1 # Number of rollout units to perform. 
```

+ Combine the two subconfigurations into the main configuration file. 


---

## 📚 3. Pydantic - Your Type Safety Guardian

Take a look at the file `src/config_factory/config_basemodel.py`. Using the pydantic documentation 
try to understand the different components of the file. 

### 🧩 **Exercise 2: Spot the Validation Magic**

> **💡 Info Point - Pydantic BaseModel**: A Python library that provides runtime type checking and data validation using Python type annotations. It converts input data to the correct types and raises clear errors for invalid data.

**📢 Action Points**:  

+ Implement an EnvConfigBasemodel class that holds and validates the environment config. 

+ Implement a RolloutConfigBasemodel class that holds and validates the rollout config. 

+ Implement different validators for each class. 

+ Make sure to explicit connect each of your yaml files with its specified config (hint: check the main config file).

---

## 📚 4. Hydra Instantiation

Now that we set our playground, we want to test our configuration setup. There different ways of using hydra. Here, we would like to test two of them: using the hydra decorator, or loading a hydra config through a function. 

**📢 Action Points**:  

+ Write a script that loads the hydra configuration using a decorator. 

+ Implement `load_and_instantiate_hydra_config(config_path, config_name)` that loads the config and instantiate it. 

+ Think about the difference between the two methods. 

+ Loading hydra configs usually generates many useless logs. Find a way how to get rid of these logs. 

> **💡 Info Point - Loading hydra**: On the one hand, using the hydra decorator is usually good when writing standalone scripts or experiment entry points. It's simple and clean, but less flexible when it comes to programmatically composing multiple configs. Besides, you can only call the decorated function once per process since hydra initializes a global state. On the other hand, using manual loading function gives you more control over the loading, is flexible and programmatic. However, it's more verbose than the decorator style.

---

## 📚 5. Rollout with Robotic Arm

In the remainder of the tutorial, we choose to stick with the functional hydra config instantiation. In this section, we want to use the `MujocoFetchReachEnv` class from `gymnasium_robotics`. We want to wrap it with the gymnasium `TimeLimit` wrapper and run a rollout with a random agent that samples an action from the action space. 

**📢 Action Points**:  

+ Implement the rollout script. 

+ In case you encounter issues with environment variables related to mujoco, find out how to fix them. How can you make the change persistent? 

+ Try different render modes, rewards. What does the dense reward computes? 

+ Implement a mechanism that computes the steps per second (when the `render_mode` is None).


