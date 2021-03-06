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

import os
import sys
import numpy as np
from py4j.java_gateway import JavaGateway

ignite_home = os.environ['IGNITE_HOME']

libs_jar = []
for f in os.listdir(ignite_home + '/libs'):
    if f.endswith('.jar'):
        libs_jar.append(ignite_home + '/libs/' + f)    
    if os.path.isdir(ignite_home + '/libs/' + f):
        for fi in os.listdir(ignite_home + '/libs/' + f):
            if fi.endswith('.jar'):
                libs_jar.append(ignite_home + '/libs/' + f + '/' + fi)

optional_libs_jar = []
for opt in os.listdir(ignite_home + '/libs/optional'):
    for f in os.listdir(ignite_home + '/libs/optional/' + opt):
        if f.endswith('.jar'):
            optional_libs_jar.append(ignite_home + '/libs/optional/' + opt + '/' + f)

classpath = ':'.join(libs_jar + optional_libs_jar)

gateway = JavaGateway.launch_gateway(classpath=classpath, die_on_exit=True)

class Utils:
    def to_java_double_array(array):
        array = np.array(array)
        java_array = gateway.new_array(gateway.jvm.double, *array.shape)
        Utils.__to_java_double_array_backtrack(array, java_array)
        return java_array

    def from_java_double_array(java_array):
        array = np.zeros(len(java_array))
        for i in range(len(java_array)):
            array[i] = java_array[i]
        return array

    def __to_java_double_array_backtrack(array, java_array):
        if array.ndim == 0:
            raise Exception("Array is scalar [dim=%d]" % array.ndim)

        for i in range(array.shape[0]):
            if array.ndim == 1:
                if array[i] is not None:
                    java_array[i] = float(array[i])
                else:
                    java_array[i] = float('NaN')
            else:
                Utils.__to_java_double_array_backtrack(array[i], java_array[i])

class Proxy:
    """Proxy class for Java object.
    """
    def __init__(self, proxy):
        """Constructs a new instance of proxy class for Java object.
        """
        self.proxy = proxy

    def proxy_or_none(proxy):
        if proxy:
            return proxy.proxy
        else:
            return None

class MLPArchitecture(Proxy):
    """MLP architecture.
    """
    def __init__(self, input_size):
        """Constructs a new instance of MLP architecture.

        Parameters
        ----------
        input_size : Input size.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.nn.architecture.MLPArchitecture(input_size)
        Proxy.__init__(self, proxy)

    def with_layer(self, neurons, has_bias=True, activator='sigmoid'):
        """Add layer.

        Parameters
        ----------
        neurons : Number of neurons.
        has_bias : Has bias or not (default value is True).
        activator : Activation function ('sigmoid', 'relu' or 'linear', default value is 'sigmoid')
        """
        java_activator = None
        if activator == 'sigmoid':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.SIGMOID
        elif activator == 'relu':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.RELU
        elif activator == 'linear':
            java_activator = gateway.jvm.org.apache.ignite.ml.nn.Activators.LINEAR
        else:
            raise Exception("Unknown activator: %s" % activator)

        self.proxy = self.proxy.withAddedLayer(neurons, has_bias, java_activator)

        return self

class LearningEnvironmentBuilder(Proxy):

    def __init__(self):
        java_proxy = gateway.jvm.org.apache.ignite.ml.environment.LearningEnvironmentBuilder.defaultBuilder()
        Proxy.__init__(self, java_proxy)

class SupervisedTrainer:
    """Supervised trainer.
    """
    @abstractmethod
    def fit(self, X, y, preprocessor=None):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        y : y.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, cache, columns, preprocessor=None):
        """Trains model based on data.

        Parameters
        ----------
        cache : Apache Ignite cache.
        columns : List of columns.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update(self, mdl, X, y, preprocessor=None):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        X : x.
        y : y.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update_on_cache(self, mdl, cache, columns, preprocessor=None):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        cache : Apache Ignite cache.
        columns : List of columns.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

class UnsupervisedTrainer:
    """Unsupervised trainer.
    """
    @abstractmethod
    def fit(self, X, preprocessor=None):
        """Trains model based on data.

        Parameters
        ----------
        X : x.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def fit_on_cache(self, data, columns, preprocessor=None):
        """Trains model based on data.

        Parameters
        ----------
        data : Apache Ignite cache.
        columns : List of columns.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update(self, mdl, X, preprocessor=None):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        X : x.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def update_on_cache(self, mdl, cache, columns, preprocessor=None):
        """Updates the model.

        Parameters
        ----------
        mdl : Model.
        cache : Apache Ignite cache.
        columns : List of columns.
        preprocessor : Preprocessor.
        """
        raise Exception("Not implemented")
