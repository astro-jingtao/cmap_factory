import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from networkx.algorithms.approximation import christofides
from scipy.spatial.distance import cdist
from skimage import color
from sklearn.cluster import DBSCAN


def distance_matrix_to_graph(distance_matrix,
                             virtual_node=False,
                             only_positive=True,
                             threshold=np.inf,
                             same_weight=False):
    """
    Convert a distance matrix to a graph.
    """

    G = nx.Graph()

    if virtual_node:
        G.add_node('v')

    num_nodes = distance_matrix.shape[0]

    for i in range(num_nodes):
        G.add_node(i)
        if virtual_node:
            G.add_edge('v', i, weight=0)
        for j in range(i + 1, num_nodes):
            if only_positive and distance_matrix[i, j] <= 0:
                continue
            if distance_matrix[i, j] > threshold:
                continue
            if same_weight:
                G.add_edge(i, j, weight=1)
            else:
                G.add_edge(i, j, weight=distance_matrix[i, j])

    return G


class ColorContainer:

    def __init__(self, colors, space='rgb', **kwargs):
        self.set_colors(colors, space=space, **kwargs)

    @property
    def colors(self):
        return self.rgb_colors

    @property
    def rgb_colors(self):
        return self._rgb_colors

    @property
    def hsv_colors(self):
        return self._hsv_colors

    @property
    def lab_colors(self):
        return self._lab_colors

    def set_colors(self, colors, space='rgb'):
        if space == 'rgb':
            # if array of int
            if np.issubdtype(colors.dtype, np.integer):
                colors = colors.astype(float) / 255.0
            self._rgb_colors = colors
            self._hsv_colors = color.rgb2hsv(colors)
            self._lab_colors = color.rgb2lab(colors)
        elif space == 'hsv':
            self._hsv_colors = colors
            self._rgb_colors = color.hsv2rgb(colors)
            self._lab_colors = color.rgb2lab(self._rgb_colors)
        elif space == 'lab':
            self._lab_colors = colors
            self._rgb_colors = color.lab2rgb(colors)
            self._hsv_colors = color.rgb2hsv(self._rgb_colors)
        else:
            raise ValueError('Invalid color space.')

    def get_distance_matrix(self, space):
        used_colors = {
            'rgb': self.rgb_colors,
            'hsv': self.hsv_colors,
            'lab': self.lab_colors
        }[space]

        return cdist(used_colors, used_colors)

    def __add__(self, other):
        return self.__class__(np.vstack((self.colors, other.colors)),
                              space='rgb')

    def __getitem__(self, idx):
        return self.__class__(self.colors[idx], space='rgb')


class ColorPool(ColorContainer):

    def __init__(self, colors, space='rgb', names=None):
        super().__init__(colors, space=space, names=names)

    def set_colors(self, colors, space='rgb', names=None):

        super().set_colors(colors, space=space)

        if names is None:
            # use hex code as name
            names = [
                f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}'
                for c in self.colors
            ]

        self.names = []

        for n in names:
            if n in self.names:
                print(f'Warning: name "{n}" already exists.')
                ii = 1
                _n = f'{n}_{ii}'
                while _n in self.names:
                    ii += 1
                    _n = f'{n}_{ii}'
                n = _n
                print(f'Using "{n}" as name.')

            self.names.append(n)

    def name_to_idx(self, name):
        return self.names.index(name)

    def get_color_list(self,
                       idx_0=None,
                       idx_1=None,
                       n_0=None,
                       n_1=None,
                       space='lab',
                       **kwargs):
        
        idx_0 = idx_0 or self.name_to_idx(n_0)
        idx_1 = idx_1 or self.name_to_idx(n_1)

        G = distance_matrix_to_graph(self.get_distance_matrix(space), **kwargs)
        _sort = nx.shortest_path(G, idx_0, idx_1)
        return ColorList(self.colors[_sort], space='rgb')

    def sort(self):
        ...

    def __add__(self, other):
        return self.__class__(np.vstack((self.colors, other.colors)),
                              space='rgb',
                              names=self.names + other.names)

    def __getitem__(self, idx):
        return self.__class__(self.colors[idx],
                              space='rgb',
                              names=self.names[idx])

    def _repr_html_(self):
        # for jupyter notebook
        return ColorList(self.colors,
                         space='rgb').to_cmap(name='color pool')._repr_html_()


class ColorList(ColorContainer):

    @property
    def rgb_colors(self):
        return self._rgb_colors

    @rgb_colors.setter
    def rgb_colors(self, value):
        self.set_colors(value, space='rgb')

    @property
    def hsv_colors(self):
        return self._hsv_colors

    @hsv_colors.setter
    def hsv_colors(self, value):
        self.set_colors(value, space='hsv')

    @property
    def lab_colors(self):
        return self._lab_colors

    @lab_colors.setter
    def lab_colors(self, value):
        self.set_colors(value, space='lab')

    def sort(self, method, tsp_space='rgb'):
        if method == 'hue':
            self.hsv_colors = sorted(self.hsv_colors, key=lambda x: x[0])
        elif method == 'tsp':
            self._sort_tsp(tsp_space)
        else:
            raise ValueError('Invalid sorting method.')

        return self

    def _sort_tsp(self, space):

        distance_matrix = self.get_distance_matrix(space)
        G = distance_matrix_to_graph(distance_matrix, virtual_node=True)
        _sort = christofides(G)[1:-1]  # remove virtual node

        self.rgb_colors = self.rgb_colors[_sort]

    def cluster(self, space='lab', return_color_lists=False, **kwargs):
        used_colors = {
            'rgb': self.rgb_colors,
            'hsv': self.hsv_colors,
            'lab': self.lab_colors
        }[space]
        clustering = DBSCAN(**kwargs).fit(used_colors)

        if return_color_lists:
            new_color_lists = {}
            for i in np.unique(clustering.labels_):
                new_color_lists[i] = ColorList(
                    self.colors[clustering.labels_ == i], space='rgb')
            return new_color_lists

        return clustering.labels_

    def to_cmap(self, mode='listed', name='custom'):
        if mode == 'listed':
            return ListedColormap(self.colors, name=name)
        elif mode == 'linear':
            return LinearSegmentedColormap.from_list(name, self.colors)
        else:
            raise ValueError('Invalid colormap mode.')

    def __repr__(self):
        return self.to_cmap(name='color list').__repr__()

    def _repr_html_(self):
        # for jupyter notebook
        return self.to_cmap(name='color list')._repr_html_()
