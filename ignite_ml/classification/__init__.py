# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Classification trainers.
"""

from ..common import SupervisedTrainer
from ..common import IgniteBiFunction

class ClassificationTrainer(SupervisedTrainer):
    """Classification trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of classification trainer.
        """
        self.proxy = proxy

    def fit(self, data, feature_extractor, label_extractor):
        """Trains a model on the given data.
        """
        return self.proxy.fit(data,
                              IgniteBiFunction(feature_extractor), 
                              IgniteBiFunction(label_extractor))

    def update(self, mdl, data, feature_extractor, label_extractor):
        """Trains the model on the given data.
        """
        return self.proxy.update(mdl,
                                 data, 
                                 IgniteBiFunction(feature_extractor), 
                                 IgniteBiFunction(label_extractor))

class ANNClassificationTrainer(ClassificationTrainer):
    """ANN classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None, k=None,
                 max_iter=None, epsilon=None, distance=None):
        """Constructs a new instance of ANN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        k : Number of clusters.
        max_iter : Max number of iterations.
        epsilon : Epsilon, delta of convergence.
        distance : Distance measure.
        """
        ClassificationTrainer.__init__(None)

class DecisionTreeClassificationTrainer(ClassificationTrainer):
    """DecisionTree classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None, max_deep=None,
                 min_impurity_decrease=None, compressor=None, use_index=True):
        """Constructs a new instance of DecisionTree classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        use_index : Use index.
        """
        ClassificationTrainer.__init__(None)

class KNNClassificationTrainer(ClassificationTrainer):
    """KNN classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None):
        """Constructs a new instance of KNN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        """
        ClassificationTrainer.__init__(None)

class LogRegClassificationTrainer(ClassificationTrainer):
    """LogisticRegression classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None, max_iter=None,
                 batch_size=None, max_loc_iter=None, update_strategy=None,
                 seed=None):
        """Constructs a new instance of LogisticRegression classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        max_loc_iter : Max number of local iterations.
        update_strategy : Update strategy.
        seed : Seed.
        """
        ClassificationTrainer.__init__(None)

class RandomForestClassificationTrainer(ClassificationTrainer):
    """RandomForest classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None,
                 trees=None, sub_sample_size=None, max_depth=None,
                 min_impurity_delta=None, features_count_selection_strategy=None,
                 nodes_to_learn_selection_strategy=None, seed=None):
        """Constructs a new instance of RandomForest classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        features_count_selection_strategy : Features count selection strategy.
        nodes_to_learn_selection_strategy : Nodes to learn selection strategy.
        seed : Seed.
        """
        ClassificationTrainer.__init__(None)

class MLPClassificationTrainer(ClassificationTrainer):
    """MLP classification trainer.
    """
    def __init__(self, env_builder=None, label_converter=None, arch, loss,
                 update_strategy, max_iter, batch_size, max_loc_iter, seed):
        """Constructs a new instance of MLP classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        arch : Architecture.
        loss : Loss function.
        update_strategy : Update strategy.
        max_iter : Max number of iterations.
        batch_size: Batch size.
        max_loc_iter: Max number of local iterations.
        seed : Seed.
        """
        ClassificationTrainer.__init__(None)
