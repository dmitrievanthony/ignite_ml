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

import numpy as np

from ..common import SupervisedTrainer
from ..common import Proxy
from ..common import Utils
from ..common import LearningEnvironmentBuilder

from ..common import gateway

class ClassificationModel(Proxy):
    """Classification model.
    """
    def __init__(self, proxy):
        """Constructs a new instance of classification model.

        Parameters
        ----------
        proxy : Proxy object that represents Java model.
        """
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

        predictions = np.array([self.__predict(x) for x in X])
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = np.hstack(predictions)

        return predictions

    def __predict(self, X):
        java_array = Utils.java_double_array(X)
        java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils
        return self.proxy.predict(java_vector_utils.of(java_array))

class ClassificationTrainer(SupervisedTrainer, Proxy):
    """Classification trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of classification trainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, X, y, preprocessor=None):
        X_java = Utils.java_double_array(X)
        y_java = Utils.java_double_array(y)

        java_model = self.proxy.fit(X_java, y_java, Proxy.proxy_or_none(preprocessor))

        return ClassificationModel(java_model)

class ANNClassificationTrainer(ClassificationTrainer):
    """ANN classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), k=2,
                 max_iter=10, eps=1e-4, distance='euclidean'):
        """Constructs a new instance of ANN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        k : Number of clusters.
        max_iter : Max number of iterations.
        eps : Epsilon, delta of convergence.
        distance : Distance measure ('euclidean', 'hamming', 'manhattan').
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.ann.ANNClassificationTrainer()

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withK(k)
        proxy.withMaxIterations(max_iter)
        proxy.withEpsilon(eps)

        java_distance = None
        if distance == 'euclidean':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.EuclideanDistance()
        elif distance == 'hamming':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.HammingDistance()
        elif distance == 'manhattan':
            java_distance = gateway.jvm.org.apache.ignite.ml.math.distances.ManhattanDistance()
        elif distance:
            raise Exception("Unknown distance type: %s" % distance)
        proxy.withDistance(java_distance)

        ClassificationTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class DecisionTreeClassificationTrainer(ClassificationTrainer):
    """DecisionTree classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), max_deep=5,
                 min_impurity_decrease=0.0, compressor=None, use_index=True):
        """Constructs a new instance of DecisionTree classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_deep : Max deep.
        min_impurity_decrease : Min impurity decrease.
        compressor : Compressor.
        use_index : Use index.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.tree.DecisionTreeClassificationTrainer(max_deep, min_impurity_decrease, compressor)

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withUseIndex(use_index)

        ClassificationTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class KNNClassificationTrainer(ClassificationTrainer):
    """KNN classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder()):
        """Constructs a new instance of KNN classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.knn.classification.KNNClassificationTrainer()
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        ClassificationTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class LogRegClassificationTrainer(ClassificationTrainer):
    """LogisticRegression classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), max_iter=100,
                 batch_size=100, max_loc_iter=100, seed=1234):
        """Constructs a new instance of LogisticRegression classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        max_loc_iter : Max number of local iterations.
        update_strategy : Update strategy.
        seed : Seed.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.regressions.logistic.LogisticRegressionSGDTrainer()

        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))

        proxy.withMaxIterations(max_iter)
        proxy.withBatchSize(batch_size)
        proxy.withLocIterations(max_loc_iter)
        if seed is not None:
            proxy.withSeed(seed)

        ClassificationTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class RandomForestClassificationTrainer(ClassificationTrainer):
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

        ClassificationTrainer.__init__(self, None)

    def fit(self, X, y):
        metas = gateway.jvm.java.util.ArrayList()
        for i in range(len(X[0])):
            meta = gateway.jvm.org.apache.ignite.ml.dataset.feature.FeatureMeta(None, i, False)
            metas.add(meta)

        self.proxy = gateway.jvm.org.apache.ignite.ml.tree.randomforest.RandomForestClassifierTrainer(metas)
        self.proxy.withEnvironmentBuilder(self.env_builder.proxy)
        self.proxy.withAmountOfTrees(self.trees)
        self.proxy.withSubSampleSize(self.sub_sample_size)
        self.proxy.withMaxDepth(self.max_depth)
        self.proxy.withMinImpurityDelta(self.min_impurity_delta)
        if self.seed is not None:
            self.proxy.withSeed(self.seed)

        self.proxy = gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(self.proxy)

        return super(RandomForestClassificationTrainer, self).fit(X, y)

class SVMClassificationTrainer(ClassificationTrainer):
    """SVM classification trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), l=0.4, max_iter=200, max_local_iter=100, seed=1234):
        """Constructs a new instance of SVM classification trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        l : Lambda.
        max_iter : Max number of iterations.
        max_loc_iter : Max number of local iterations.
        seed : Seed.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.svm.SVMLinearClassificationTrainer()    
        proxy.withEnvironmentBuilder(Proxy.proxy_or_none(env_builder))
        proxy.withLambda(l)
        proxy.withAmountOfIterations(max_iter)
        proxy.withAmountOfLocIterations(max_local_iter)
        if seed is not None:
            proxy.withSeed(seed)

        ClassificationTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(proxy))

class MLPClassificationTrainer(ClassificationTrainer):
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
