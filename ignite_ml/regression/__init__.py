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

class RegressionModel(Proxy):

    def __init__(self, proxy, accepts_matrix):
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

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
            if X[i] is not None:
                java_array[i] = float(X[i])
            else:
                java_array[i] = float('NaN')
        if self.accepts_matrix:
            return self.proxy.predict(java_vector_utils.of(java_array).toMatrix(True)).get(0, 0)
        else:
            return self.proxy.predict(java_vector_utils.of(java_array))

class RegressionTrainer(SupervisedTrainer, Proxy):
    """Regression.
    """
    def __init__(self, proxy, accepts_matrix=False):
        """Constructs a new instance of regression trainer.
        """
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

    def fit(self, X, y, preprocessor=None):
        X_java = gateway.new_array(gateway.jvm.double, len(X), len(X[0]))
        y_java = gateway.new_array(gateway.jvm.double, len(y))

        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] is not None:
                    X_java[i][j] = float(X[i][j])
                else:
                    X_java[i][j] = float('NaN')
            if y[i] is not None:
                y_java[i] = float(y[i])
            else:
                y_java[i] = float('NaN')

        java_model = self.proxy.fit(X_java, y_java, Proxy.proxy_or_none(preprocessor))
    
        return RegressionModel(java_model, self.accepts_matrix)

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
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withUsingIdx(use_index)

        RegressionTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

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
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        RegressionTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

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
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        RegressionTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class RandomForestRegressionTrainer(RegressionTrainer):
    """RandomForest classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(),
                 trees=1, sub_sample_size=1.0, max_depth=5,
                 min_impurity_delta=0.0, features_count_selection_strategy=None,
                 nodes_to_learn_selection_strategy=None, seed=None):
        """Constructs a new instance of RandomForest classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        features_count_selection_strategy : Features count selection strategy.
        nodes_to_learn_selection_strategy : Nodes to learn selection strategy.
        seed : Seed.
        """
        self.env_builder = env_builder
        self.trees = trees
        self.sub_sample_size = sub_sample_size
        self.max_depth = max_depth
        self.min_impurity_delta = min_impurity_delta
        self.features_count_selection_strategy = features_count_selection_strategy
        self.nodes_to_learn_selection_strategy = nodes_to_learn_selection_strategy
        self.seed = seed

        RegressionTrainer.__init__(self, None)

    def fit(self, X, y):
        metas = gateway.jvm.java.util.ArrayList()
        for i in range(len(X[0])):
            meta = gateway.jvm.org.apache.ignite.ml.dataset.feature.FeatureMeta(None, i, False)
            metas.add(meta)

        self.proxy = gateway.jvm.org.apache.ignite.ml.tree.randomforest.RandomForestRegressionTrainer(metas)
        self.proxy.withEnvironmentBuilder(self.env_builder.proxy)
        self.proxy.withAmountOfTrees(self.trees)
        self.proxy.withSubSampleSize(self.sub_sample_size)
        self.proxy.withMaxDepth(self.max_depth)
        self.proxy.withMinImpurityDelta(self.min_impurity_delta)
        #self.proxy.withFeatureCountSelectionStrategy(self.feature_count_selection_strategy)
        #self.proxy.withNodesToLearnSelectionStrategy(self.nodes_to_learn_selection_strategy)
        if self.seed:
            self.proxy.withSeed(self.seed)

        self.proxy = gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(self.proxy)

        return super(RandomForestRegressionTrainer, self).fit(X, y)

class MLPRegressionTrainer(RegressionTrainer):
    """MLP regression trainer.
    """
    def __init__(self, arch, env_builder=LearningEnvironmentBuilder(), loss='mse',
                 learning_rate=0.1, max_iter=1000, batch_size=100, loc_iter=10, seed=None):
        """Constructs a new instance of MLP regression trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        arch : Architecture.
        loss : Loss function ('mse', 'log', 'l2', 'l1' or 'hinge', default value is 'mse').
        update_strategy : Update strategy.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        loc_iter : Number of local iterations.
        seed : Seed.
        """
        java_loss = None
        if loss == 'mse':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.MSE
        elif loss == 'log':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.LOG
        elif loss == 'l2':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.L2
        elif loss == 'l1':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.L1
        elif loss == 'hinge':
            java_loss = gateway.jvm.org.apache.ignite.ml.optimization.LossFunctions.HINGE
        else:
            raise Exception('Unknown loss: %s' % loss)

        proxy = gateway.jvm.org.apache.ignite.ml.python.PythonMLPDatasetTrainer(arch.proxy, java_loss, learning_rate, max_iter, batch_size, loc_iter, seed)
        RegressionTrainer.__init__(self, proxy, True)
