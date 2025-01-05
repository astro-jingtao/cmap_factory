import json
import os

import numpy as np

from .factory import ColorPool

# get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def get_preset_color_pool(name, use_pinyin=False):
    if name == 'chinese':
        with open(os.path.join(current_dir, 'data/chinese_color.json'),
                  'r',
                  encoding='utf-8') as f:
            chinese_color = json.load(f)
        colors = []
        names = []
        for c in chinese_color:
            colors.append(c['RGB'])
            if use_pinyin:
                names.append(c['pinyin'])
            else:
                names.append(c['name'])
        colors = np.array(colors)
        return ColorPool(colors, names=names, space='rgb')
    else:
        raise ValueError('Invalid preset color pool name')
