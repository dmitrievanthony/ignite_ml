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

"""Clusterer.
"""

import numpy as np

from ..common import UnsupervisedTrainer
from ..common import Proxy
from ..common import EuclideanDistance
from ..common import LearningEnvironmentBuilder

from ..common import gateway

class ClusteringModel:

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

class ClusteringTrainer(UnsupervisedTrainer, Proxy):
    """Clustering trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of ClusteringTrainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, X):
        X_java = gateway.new_array(gateway.jvm.double, len(X), len(X[0]))
        y_java = gateway.new_array(gateway.jvm.double, len(X))

        for i in range(len(X)):
            for j in range(len(X[i])):
                X_java[i][j] = float(X[i][j])

        java_model = gateway.jvm.org.apache.ignite.ml.python.PythonDatasetTrainer(self.proxy).fit(X_java, y_java)

        return ClusteringModel(java_model)

class GMMClusteringTrainer(ClusteringTrainer):
    """GMM clustring trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), eps=1e-3, count_of_components=2,
                 max_iter=10, initial_means=None, max_count_of_init_tries=3, max_count_of_clusters=2,
                 max_likelihood_divirgence=5.0, min_elements_for_new_cluster=300, min_cluster_probability=0.05):
        """Constructs a new instance of GMM clustring trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        count_of_components : Count of components.
        max_iter : Max number of iterations.
        initial_means : Initial means.
        max_count_of_init_tries : Max count of init tries.
        max_count_of_clusters : Max count of clusters.
        max_likelihood_divirgence : Max likelihood divirgence.
        min_elements_for_new_cluster : Min elements for new cluster.
        min_cluster_probability : Min cluster probability.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.clustering.gmm.GmmTrainer()
        proxy.withEnvironmentBuilder(env_builder.proxy)
        proxy.withInitialCountOfComponents(count_of_components)
        #proxy.withInitialMeans(initial_means)
        proxy.withMaxCountIterations(max_iter)
        proxy.withEps(eps)
        proxy.withMaxCountOfInitTries(max_count_of_init_tries)
        proxy.withMaxCountOfClusters(max_count_of_clusters)
        proxy.withMaxLikelihoodDivergence(max_likelihood_divirgence)
        proxy.withMinElementsForNewCluster(min_elements_for_new_cluster)
        proxy.withMinClusterProbability(min_cluster_probability)

        ClusteringTrainer.__init__(self, proxy)

class KMeansClusteringTrainer(ClusteringTrainer):
    """KMeans clustring trainer.
    """
    def __init__(self, env_builder=LearningEnvironmentBuilder(), amount_of_clusters=2,
                 max_iter=10, eps=1e-4, distance=EuclideanDistance()):
        """Constructs a new instance of KMeans clustering trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        amount_of_clusters : Amount of clusters.
        max_iter : Max number of iterations.
        eps : Epsilon.
        distance : Distance measure.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.clustering.kmeans.KMeansTrainer()
        proxy.withEnvironmentBuilder(env_builder.proxy)
        proxy.withAmountOfClusters(amount_of_clusters)
        proxy.withMaxIterations(max_iter)
        proxy.withEpsilon(eps)
        proxy.withDistance(distance.proxy)

        ClusteringTrainer.__init__(self, proxy)
