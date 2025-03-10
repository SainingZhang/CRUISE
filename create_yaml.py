import yaml
from pathlib import Path

class CustomDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, *args, **kwargs):
        return super().increase_indent(flow=flow, indentless=False)

def represent_list(self, data):
    """Custom representer for lists to use flow style (brackets)"""
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def represent_str(self, data):
    """Custom representer to force quotes for specific strings"""
    if any(data.startswith(prefix) for prefix in ['/mnt/']):
        style = '"'
    else:
        style = None
    return self.represent_scalar('tag:yaml.org,2002:str', data, style=style)

def represent_dict(self, data):
    """Custom representer for dictionaries to use flow style for selected_frames"""
    if any(isinstance(v, list) for v in data.values()):  # Check if dict contains lists
        return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)
    return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)

# Register the custom representers
CustomDumper.add_representer(list, represent_list)
CustomDumper.add_representer(str, represent_str)
CustomDumper.add_representer(dict, represent_dict)

def generate_config(sequence_id):
    seq_str = f"{sequence_id:04d}"  # 6 digits for path
    
    config = {
        'task': f'dair_seq_{seq_str}',
        'source_path': [f'/mnt/zhangsn/data/V2X-Seq-SPD-Processed/{seq_str}_0_original'],
        'exp_name': 'exp_1_comp',
        'gpus': [1],
        'specified_sequence_id': [seq_str],
        'resolution': 1,
        
        'data': {
            'split_test': -1,
            'split_train': 1,
            'type': 'Dair',
            'white_background': False,
            'selected_frames': {
                seq_str: [0, 143]
            },
            'cameras': [0, 1],
            'extent': 10,
            'use_colmap': False,
            'filter_colmap': False,
            'box_scale': 1.0,
            'use_mono_normal': False,
            'use_mono_depth': False
        }
    }
    
    return config

def save_yaml(config, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)

def generate_multiple_configs(sequence_ids, output_dir='configs/seqs'):
    for seq_id in sequence_ids:
        config = generate_config(seq_id)
        filename = f"{output_dir}/{seq_id:04d}_comp.yaml"
        save_yaml(config, filename)
        print(f"Generated config file: {filename}")


if __name__ == "__main__":
    # List of sequence numbers to generate configs for
    sequence_ids = [1]  # Will be formatted as 0008, 0009, 0010, 0011
    
    # Generate configs for all sequence IDs
    generate_multiple_configs(sequence_ids)