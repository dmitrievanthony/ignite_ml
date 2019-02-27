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

"""Regression trainers.
"""

import numpy as np

from ..common import SupervisedTrainer
from ..common import Proxy
from ..common import LearningEnvironmentBuilder

from ..common import gateway

class RegressionModel:

    def __init__(self, proxy):
        self.proxy = proxy

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            return self.__predict(X)
        elif len(X.shape) == 2:
            return [self.__predict(x) for x in X]

    def __predict(self, X):
        java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils
        java_array = gateway.new_array(gateway.jvm.double, len(X))
        for i in range(len(X)):
            java_array[i] = float(X[i])
        return self.proxy.predict(java_vector_utils.of(java_array))

class RegressionTrainer(SupervisedTrainer, Proxy):
    """Regression.
    """
    def __init__(self, proxy):
        """Constructs a new instance of regression trainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, X, y):
        X_java = gateway.new_array(gateway.jvm.double, len(X), len(X[0]))
        y_java = gateway.new_array(gateway.jvm.double, len(y))

        for i in range(len(X)):
            for j in range(len(X[i])):
                X_java[i][j] = float(X[i][j])
            y_java[i] = float(y[i])

        java_model = gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(self.proxy).fit(X_java, y_java)
    
        return RegressionModel(java_model)

class DecisionTreeRegressionTrainer(RegressionTrainer):
    """DecisionTree regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(),
                 max_deep=5,
                 min_impurity_decrease=0.0, compressor=None, use_index=True):
        """Constructs a new instance of DecisionTree regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.tree.DecisionTreeRegressionTrainer(max_deep, min_impurity_decrease, compressor)
        proxy.withEnvironmentBuilder(env_builder.proxy)
        proxy.withUsingIdx(use_index)

        RegressionTrainer.__init__(self, proxy)

class KNNRegressionTrainer(RegressionTrainer):
    """KNN regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of linear regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.regression.KNNRegressionTrainer()
        proxy.withEnvironmentBuilder(env_builder.proxy)

        RegressionTrainer.__init__(self, proxy)

class LinearRegressionTrainer(RegressionTrainer):
    """Linear regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of linear regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.regressions.linear.LinearRegressionLSQRTrainer()
        proxy.withEnvironmentBuilder(env_builder.proxy)

        RegressionTrainer.__init__(self, proxy)

class RandomForestRegressionTrainer(RegressionTrainer):
    """RandomForest regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(),
                  trees=1, sub_sample_size=1.0, max_depth=5,
                  min_impurity_delta=0.0, features_count_selection_strategy=None,
                  nodes_to_learn_selection_strategy=None, seed=None):
        """Constructs a new instance of RandomForest regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        feature_count_selection_strategy : Feature count selection strategy.
        nodes_to_learn_selection_strategy : Nodes to learn selection strategy.
        seed : Seed.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.tree.randomforest.RandomForestRegressionTrainer()
        proxy.withEnvironmentBuilder(env_builder.proxy)
        proxy.withAmountOfTrees(trees)
        proxy.withSubSampleSize(subSampleSize)
        proxy.withMaxDepth(maxDepth)
        proxy.withMinImpurityDelta(minImpurityDelta)
        #proxy.withFeatureCountSelectionStrategy(featureCountSelectionStrategy)
        #proxy.withNodesToLearnSelectionStrategy(nodesToLearnSelectionStrategy)
        proxy.withSeed(seed)

        RegressionTrainer.__init__(self, proxy)

class MLPRegressionTrainer(RegressionTrainer):
    """MLP regression trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), arch=None, loss=None,
                 update_strategy=None, max_iter=None, batch_size=None, max_loc_iter=None, seed=None):
        """Constructs a new instance of MLP regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        arch : Architecture.
        loss : Loss function.
        update_strategy : Update strategy.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        max_loc_iter : Max number of local iterations.
        seed : Seed.
        """
        RegressionTrainer.__init__(self, None)
        raise Exception("Unsupported")
