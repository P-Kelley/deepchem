import torch
from typing import Optional, Callable, Dict, List, Tuple

class GraphConv(torch.nn.Module):
    """Graph Convolutional Layers

    This layer implements the graph convolution introduced in [1]_.  The graph
    convolution combines per-node feature vectures in a nonlinear fashion with
    the feature vectors for neighboring nodes.  This "blends" information in
    local neighborhoods of a graph.

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints."
        Advances in neural information processing systems. 2015. https://arxiv.org/abs/1509.09292

  """

    def __init__(self,
                 out_channel: int,
                 min_deg: int = 0,
                 max_deg: int = 10,
                 activation_fn: Optional[Callable] = None,
                 **kwargs):
        """Initialize a graph convolutional layer.

        Parameters
        ----------
        out_channel: int
            The number of output channels per graph node.
        min_deg: int, optional (default 0)
            The minimum allowed degree for each graph node.
        max_deg: int, optional (default 10)
            The maximum allowed degree for each graph node. Note that this
            is set to 10 to handle complex molecules (some organometallic
            compounds have strange structures). If you're using this for
            non-molecular applications, you may need to set this much higher
            depending on your dataset.
        activation_fn: function
            A nonlinear activation function to apply. If you're not sure,
            `tf.nn.relu` is probably a good default for your application.
        """
        super(GraphConv, self).__init__(**kwargs)
        self.out_channel = out_channel
        self.min_degree = min_deg
        self.max_degree = max_deg
        self.activation_fn = activation_fn

    def build(self, input_shape):
        # Generate the nb_affine weights and biases
        num_deg = 2 * self.max_degree + (1 - self.min_degree)
        self.W_list = [
            self.add_weight(name='kernel' + str(k),
                            shape=(int(input_shape[0][-1]), self.out_channel),
                            initializer='glorot_uniform',
                            trainable=True) for k in range(num_deg)
        ]
        self.b_list = [
            self.add_weight(name='bias' + str(k),
                            shape=(self.out_channel, ),
                            initializer='zeros',
                            trainable=True) for k in range(num_deg)
        ]
        self.built = True

    def get_config(self):
        config = super(GraphConv, self).get_config()
        config['out_channel'] = self.out_channel
        config['min_deg'] = self.min_degree
        config['max_deg'] = self.max_degree
        config['activation_fn'] = self.activation_fn
        return config

    def call(self, inputs):

        # Extract atom_features
        atom_features = inputs[0]

        # Extract graph topology
        deg_slice = inputs[1]
        deg_adj_lists = inputs[3:]

        W = iter(self.W_list)
        b = iter(self.b_list)

        # Sum all neighbors using adjacency matrix
        deg_summed = self.sum_neigh(atom_features, deg_adj_lists)

        # Get collection of modified atom features
        new_rel_atoms_collection = (self.max_degree + 1 -
                                    self.min_degree) * [None]

        split_features = torch.split(atom_features, deg_slice[:, 1])
        for deg in range(1, self.max_degree + 1):
            # Obtain relevant atoms for this degree
            rel_atoms = deg_summed[deg - 1]

            # Get self atoms
            self_atoms = split_features[deg - self.min_degree]

            # Apply hidden affine to relevant atoms and append
            rel_out = torch.matmul(rel_atoms, next(W)) + next(b)
            self_out = torch.matmul(self_atoms, next(W)) + next(b)
            out = rel_out + self_out

            new_rel_atoms_collection[deg - self.min_degree] = out

        # Determine the min_deg=0 case
        if self.min_degree == 0:
            self_atoms = split_features[0]

            # Only use the self layer
            out = torch.matmul(self_atoms, next(W)) + next(b)

            new_rel_atoms_collection[0] = out

        # Combine all atoms back into the list
        atom_features = torch.concat(axis=0, values=new_rel_atoms_collection)

        if self.activation_fn is not None:
            atom_features = self.activation_fn(atom_features)

        return atom_features

    def sum_neigh(self, atoms, deg_adj_lists):
        """Store the summed atoms by degree"""
        deg_summed = self.max_degree * [None]

        # Tensorflow correctly processes empty lists when using concat
        for deg in range(1, self.max_degree + 1):
            gathered_atoms = torch.gather(atoms, deg_adj_lists[deg - 1])
            # Sum along neighbors as well as self, and store
            summed_atoms = torch.mean(gathered_atoms, 1)
            deg_summed[deg - 1] = summed_atoms

        return deg_summed


class GraphPool(torch.nn.Module):
    """A GraphPool gathers data from local neighborhoods of a graph.

    This layer does a max-pooling over the feature vectors of atoms in a
    neighborhood. You can think of this layer as analogous to a max-pooling
    layer for 2D convolutions but which operates on graphs instead. This
    technique is described in [1]_.

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015. https://arxiv.org/abs/1509.09292

    """

    def __init__(self, min_degree=0, max_degree=10, **kwargs):
        """Initialize this layer

        Parameters
        ----------
        min_deg: int, optional (default 0)
            The minimum allowed degree for each graph node.
        max_deg: int, optional (default 10)
            The maximum allowed degree for each graph node. Note that this
            is set to 10 to handle complex molecules (some organometallic
            compounds have strange structures). If you're using this for
            non-molecular applications, you may need to set this much higher
            depending on your dataset.
        """
        super(GraphPool, self).__init__(**kwargs)
        self.min_degree = min_degree
        self.max_degree = max_degree

    def get_config(self):
        config = super(GraphPool, self).get_config()
        config['min_degree'] = self.min_degree
        config['max_degree'] = self.max_degree
        return config

    def call(self, inputs):
        atom_features = inputs[0]
        deg_slice = inputs[1]
        deg_adj_lists = inputs[3:]

        # Perform the mol gather
        # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
        #                            self.max_degree, self.min_degree)

        deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

        # Tensorflow correctly processes empty lists when using concat

        split_features = torch.split(atom_features, deg_slice[:, 1])
        for deg in range(1, self.max_degree + 1):
            # Get self atoms
            self_atoms = split_features[deg - self.min_degree]

            if deg_adj_lists[deg - 1].shape[0] == 0:
                # There are no neighbors of this degree, so just create an empty tensor directly.
                maxed_atoms = torch.zeros((0, self_atoms.shape[-1]))
            else:
                # Expand dims
                self_atoms = torch.unsqueeze(self_atoms, 1)

                # always deg-1 for deg_adj_lists
                gathered_atoms = torch.gather(atom_features,
                                              deg_adj_lists[deg - 1])
                gathered_atoms = torch.concat(
                    axis=1, values=[self_atoms, gathered_atoms])

                maxed_atoms = torch.max(gathered_atoms, 1)
            deg_maxed[deg - self.min_degree] = maxed_atoms

        if self.min_degree == 0:
            self_atoms = split_features[0]
            deg_maxed[0] = self_atoms

        return torch.concat(axis=0, values=deg_maxed)


class GraphGather(torch.nn.Module):
    """A GraphGather layer pools node-level feature vectors to create a graph feature vector.

    Many graph convolutional networks manipulate feature vectors per
    graph-node. For a molecule for example, each node might represent an
    atom, and the network would manipulate atomic feature vectors that
    summarize the local chemistry of the atom. However, at the end of
    the application, we will likely want to work with a molecule level
    feature representation. The `GraphGather` layer creates a graph level
    feature vector by combining all the node-level feature vectors.

    One subtlety about this layer is that it depends on the
    `batch_size`. This is done for internal implementation reasons. The
    `GraphConv`, and `GraphPool` layers pool all nodes from all graphs
    in a batch that's being processed. The `GraphGather` reassembles
    these jumbled node feature vectors into per-graph feature vectors.

    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015. https://arxiv.org/abs/1509.09292
    """

    def __init__(self, batch_size, activation_fn=None, **kwargs):
        """Initialize this layer.

        Parameters
        ---------
        batch_size: int
            The batch size for this layer. Note that the layer's behavior
            changes depending on the batch size.
        activation_fn: function
            A nonlinear activation function to apply. If you're not sure,
            `tf.nn.relu` is probably a good default for your application.
        """

        super(GraphGather, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.activation_fn = activation_fn

    def get_config(self):
        config = super(GraphGather, self).get_config()
        config['batch_size'] = self.batch_size
        config['activation_fn'] = self.activation_fn
        return config

    def call(self, inputs):
        """Invoking this layer.

        Parameters
        ----------
        inputs: list
            This list should consist of `inputs = [atom_features, deg_slice,
            membership, deg_adj_list placeholders...]`. These are all
            tensors that are created/process by `GraphConv` and `GraphPool`
        """
        atom_features = inputs[0]

        # Extract graph topology
        membership = inputs[2]

        assert self.batch_size > 1, "graph_gather requires batches larger than 1"

        sparse_reps = tf.math.unsorted_segment_sum(atom_features, membership,
                                                   self.batch_size)
        max_reps = tf.math.unsorted_segment_max(atom_features, membership,
                                                self.batch_size)
        mol_features = torch.concat(axis=1, values=[sparse_reps, max_reps])

        if self.activation_fn is not None:
            mol_features = self.activation_fn(mol_features)
        return mol_features

#tf.math.unsorted_segment_sum and tf.math.unsorted_segment_max do not have direct pytorch equivilents. Will have to replace these with other functions that do the same thing

#For 2d tensors this can replace tf.math.unsorted_segment_sum. Could determine batch size based off the largest index
#torch.zeros(torch.tensor(membership).shape[0], torch.max(torch.tensor(membership)) + 1).scatter_add(dim=1, index=torch.tensor(membership), src=torch.tensor(atom_features)).sum(0)

#This one does not determine batch size automatically based off the largest index. May be needed if GraphGather is called repeatedly for each batch
#torch.zeros(torch.tensor(membership).shape[0], self.batch_size).scatter_add(dim=1, index=torch.tensor(membership), src=torch.tensor(atom_features)).sum(0)