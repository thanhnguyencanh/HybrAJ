from config import *

CONFIG_TEST = {
    "0": [{'pick': ['black spoon', 'yellow lemon'],
           'place': ['blue block', 'orange bowl', 'green bowl', 'orange block']}],
    "1": [{'pick': ['yellow block', 'red apple', 'red strawberry'], 'place': ['orange block']}],
    "2": [{'pick': ['blue block', 'black knife'], 'place': ['red block', 'yellow block', 'green bowl', 'blue bowl']}],
    "3": [
        {'pick': ['yellow lemon', 'yellow banana'], 'place': ['yellow block', 'blue block']},
        {'pick': ['blue block', 'red block'], 'place': ['orange bowl', 'yellow block', 'orange block', 'green block']},
        {'pick': ['blue block', 'red block', 'yellow banana'], 'place': ['yellow bowl', 'orange bowl', 'green block']},
        {'pick': ['blue block', 'yellow block'], 'place': ['red block', 'green block']}
    ]
}

def random_env(action, seed):
    np.random.seed(seed)
    # 1. Convert action to string to match CONFIG_TEST keys
    configs = CONFIG_TEST[str(action)]

    # 2. Pick a random index (Simplest way to pick from a list of dicts)
    idx = np.random.randint(len(configs))
    config = configs[idx]
    # 3. Get instruction (assuming instruction_form exists in your config.py)
    instruction = instruction_form[action]

    return config, instruction

# print(random_env(3))
