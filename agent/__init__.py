"""
Taken verbatim from "Learning Character-Agnostic Motion for Motion Retargeting in 2D" 
https://github.com/ChrisWu1997/2D-Motion-Retargeting/tree/7eaae7e87e927d279ad91e703b6e8f8b4d482f64
"""


from agent.agents import Agent2x, Agent3x


def get_training_agent(config, net):
    assert config.name is not None
    if config.name == "skeleton" or config.name == "view":
        return Agent2x(config, net)
    else:
        return Agent3x(config, net)
