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

class Classifier(SupervisedTrainer):
    """Classifier.
    """
    pass

class ANNClassifier(Classifier):
    """ANN Classifier.
    """
    def __init__(self, env_builder=None, label_converter=None, k=None,
                 max_iter=None, epsilon=None, distance=None):
        """ANN Classifier.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        k : Number of clusters.
        max_iter : Max number of iterations.
        epsilon : Epsilon, delta of convergence.
        distance : Distance measure.
        """
        pass

class DecisionTreeClassifier(Classifier):
    """Decision Tree Classifier.
    """
    def __init__(self, env_builder=None, label_converter=None, max_deep=None,
                 min_impurity_decrease=None, compressor=None, use_index=True):
        """Decision Tree Classifier.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        use_index : Use index.
        """
        pass

class KNNClassifier(Classifier):
    """KNN Classifier.
    """
    def __init__(self, env_builder=None, label_converter=None):
        """KNN Classifier.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        """
        pass

class LogRegClassifier(Classifier):
    """Logistic Regression Classifier.
    """
    def __init__(self, env_builder=None, label_converter=None, max_iter=None,
                 batch_size=None, max_loc_iter=None, update_strategy=None,
                 seed=None):
        """Logistic Regression Classifier.

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
        pass

class RandomForestClassifier(Classifier):
    """Random Forest Classifier.
    """
    def __init__(self, env_builder=None, label_converter=None,
                 trees=None, sub_sample_size=None, max_depth=None,
                 min_impurity_delta=None, features_count_selection_strategy=None,
                 nodes_to_learn_selection_strategy=None, seed=None):
        """Random Forest Classifier.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        features_count_selection_strategy : Features count selection strategy.
        """
        pass
