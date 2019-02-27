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

"""Common classes.
"""

from abc import abstractmethod
from py4j.java_gateway import JavaGateway

gateway = JavaGateway(start_callback_server=True)
#gateway.restart_callback_server()

class Proxy:
    """Proxy class for Java object.
    """
    def __init__(self, proxy):
        """Constructs a new instance of proxy class for Java object.
        """
        self.proxy = proxy

class LearningEnvironmentBuilder(Proxy):

    def __init__(self):
        java_proxy = gateway.jvm.org.apache.ignite.ml.environment.LearningEnvironmentBuilder.defaultBuilder()
        Proxy.__init__(self, java_proxy)

class SupervisedTrainer:
    """Supervised trainer.
    """
    @abstractmethod
    def fit(self, X, y):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        y : y.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, cache, columns):
        """Trains model based on data.

        Parameters
        ----------
        cache : Apache Ignite cache.
        columns : List of columns
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update(self, mdl, X, y):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        X : x.
        y : y.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update_on_cache(self, mdl, cache, columns):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        cache : Apache Ignite cache.
        columns : List of columns.
        """
        raise Exception("Not implemented")

class UnsupervisedTrainer:
    """Unsupervised trainer.
    """
    @abstractmethod
    def fit(self, X):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, data, columns):
        """Trains model based on data.

        Parameters
        ----------
        data : Apache Ignite cache.
        columns : List of columns.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update(self, mdl, X):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        X : x.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update_on_cache(self, mdl, cache, columns):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        cache : Apache Ignite cache.
        columns : List of columns.
        """
        raise Exception("Not implemented")

class DistanceMeasure:
    pass

class EuclideanDistance(DistanceMeasure, Proxy):
    """Constructs a new instance of Euclidean distance.
    """
    def __init__(self):
       Proxy.__init__(self, gateway.jvm.org.apache.ignite.ml.math.distances.EuclideanDistance()) 
