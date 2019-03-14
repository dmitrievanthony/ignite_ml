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
from ..common import Utils
from ..common import LearningEnvironmentBuilder

from ..common import gateway

class RegressionModel(Proxy):
    """Regression model.
    """
    def __init__(self, proxy, accepts_matrix):
        """Constructs a new instance of regression model.

        Parameters
        ----------
        proxy : Proxy object that represents Java model.
        accept_matrix : Flag that identifies if model accepts matrix or vector.
        """
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

    def predict(self, X):
        """Predicts a result.

        Parameters
        ----------

        X : Features.
        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        elif X.ndim > 2:
            raise Exception("X has unexpected dimension [dim=%d]" % X.ndim)

        # Check if model accepts multiple objects for inference.
        if self.accepts_matrix:
            java_array = Utils.to_java_double_array(X)
            java_matrix = gateway.jvm.org.apache.ignite.ml.math.primitives.matrix.impl.DenseMatrix(java_array)
            # Check if model is a single model or model-per-label.
            if isinstance(self.proxy, list):
                predictions = np.array([mdl.predict(java_matrix) for mdl in self.proxy])
            else:
                res = self.proxy.predict(java_matrix)
                rows = res.rowSize()
                cols = res.columnSize()
                predictions = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        predictions[i, j] = res.get(i, j)
        else:
            predictions = []
            for i in range(X.shape[0]):
                java_array = Utils.to_java_double_array(X[i])
                java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils
                # Check if model is a single model or model-per-label.
                if isinstance(self.proxy, list):
                    prediction = [mdl.predict(java_vector_utils.of(java_array)) for mdl in self.proxy]
                else:
                    prediction = [self.proxy.predict(java_vector_utils.of(java_array))]
                predictions.append(prediction)
            predictions = np.array(predictions)

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = np.hstack(predictions)

        return predictions

class RegressionTrainer(SupervisedTrainer, Proxy):
    """Regression.
    """
    def __init__(self, proxy, multiple_labels=False, accepts_matrix=False):
        """Constructs a new instance of regression trainer.
        """
        self.multiple_labels = multiple_labels
        self.accepts_matrix = accepts_matrix
        Proxy.__init__(self, proxy)

    def fit(self, X, y, preprocessor=None):
        X = np.array(X)
        y = np.array(y)

        # Check dimensions: we expected to have 2-dim X and y arrays.
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        elif X.ndim > 2:
            raise Exception("X has unexpected dimension [dim=%d]" % X.ndim)

        if y.ndim == 1:
            y = y.reshape(y.shape[0], 1)
        elif y.ndim > 2:
            raise Exception("y has unexpected dimension [dim=%d]" % y.ndim)

        X_java = Utils.to_java_double_array(X)

        # We have two types of models: first type can accept multiple labels, second can't.
        if self.multiple_labels:        
            y_java = Utils.to_java_double_array(y)
            java_model = self.proxy.fit(X_java, y_java, Proxy.proxy_or_none(preprocessor))
            return RegressionModel(java_model, self.accepts_matrix)
        else:
            java_models = []
            # Here we need to prepare a model for each y column.
            for i in range(y.shape[1]):
                y_java = Utils.to_java_double_array(y[:,i])
                java_model = self.proxy.fit(X_java, y_java, Proxy.proxy_or_none(preprocessor))
                java_models.append(java_model)
            return RegressionModel(java_models, self.accepts_matrix)

    def fit_on_cache(self, cache, preprocessor=None):
        java_model = self.proxy.fitOnCache(cache.proxy, Proxy.proxy_or_none(preprocessor))

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
                 min_impurity_delta=0.0, seed=None):
        """Constructs a new instance of RandomForest classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        trees : Number of trees.
        sub_sample_size : Sub sample size.
        max_depth : Max depth.
        min_impurity_delta : Min impurity delta.
        seed : Seed.
        """
        self.env_builder = env_builder
        self.trees = trees
        self.sub_sample_size = sub_sample_size
        self.max_depth = max_depth
        self.min_impurity_delta = min_impurity_delta
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
        if self.seed is not None:
            self.proxy.withSeed(self.seed)

        self.proxy = gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(self.proxy)

        return super(RandomForestRegressionTrainer, self).fit(X, y)

    def fit_on_cache(self, cache, preprocessor=None):
        raise Exception("Not implemented")

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
        RegressionTrainer.__init__(self, proxy, True, True)
