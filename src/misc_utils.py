import json

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_attack_kwargs(config):
    """
    Set up PGD attack parameters based on the configuration.
    """
    if config['type'] == 'pgd':
        attack_kwargs = {
            'constraint': config['constraint'],
            'eps': config['epsilon'],
            'step_size': 2.5 * config['epsilon'] / config['iterations'], #config['step_size'],    # Modification to imitate original PGD paper
            'iterations': config['iterations'],
            'targeted': config['targeted'],
            'custom_loss': None
        }
    return attack_kwargs