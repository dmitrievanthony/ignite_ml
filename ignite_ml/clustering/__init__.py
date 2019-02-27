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

from ..common/UnsupervisedTrainer

class ClusteringTrainer(UnsupervisedTrainer, Proxy):
    """Clustering trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of ClusteringTrainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, data, feature_extractor):
        return self.proxy.fit(data,
                              IgniteBiFunction(feature_extractor))

    def update(self, mdl, data, feature_extractor):
        return self.proxy.fit(mdl,
                              data,
                              IgniteBiFunction(feature_extractor))

class GMMClusteringTrainer(ClusteringTrainer):
    """GMM clustring trainer.
    """
    def __init__(self, env_builder=None, count_of_components,
                 max_iter, initial_means, eps, max_init_tries):
        """Constructs a new instance of GMM clustring trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        count_of_components : Count of components.
        max_iter : Max number of iterations.
        initial_means : Initial means.
        eps : Epsilon.
        max_init_tries : Max init tries.
        """
        ClusteringTrainer.__init__(self, None)

class KMeansClusteringTrainer(ClusteringTrainer):
    """KMeans clustring trainer.
    """
    def __init__(self, env_builder=None, amount_of_clusters,
                 max_iter, eps, distance):
        """Constructs a new instance of KMeans clustering trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        amount_of_clusters : Amount of clusters.
        max_iter : Max number of iterations.
        eps : Epsilon.
        distance : Distance measure.
        """
        ClusteringTrainer.__init__(self, None)
