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

"""Preprocessors.
"""

import numpy as np

from ..common import UnsupervisedTrainer
from ..common import Proxy
from ..common import LearningEnvironmentBuilder

from ..common import gateway

class PreprocessingModel(Proxy):
    """Preprocessing model.
    """
    def __init__(self, proxy):
        """Constructs a new instance of preprocessing model.
        """
        Proxy.__init__(self, proxy)

    def transform(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            return self.__transform(X)
        elif len(X.shape) == 2:
            return np.array([self.__transform(x) for x in X])

    def __transform(self, X):
        java_vector_utils = gateway.jvm.org.apache.ignite.ml.math.primitives.vector.VectorUtils
        java_array = gateway.new_array(gateway.jvm.double, len(X))
        for i in range(len(X)):
            if X[i] is not None:
                java_array[i] = float(X[i])
            else:
                java_array[i] = float('NaN')
        res = self.proxy.apply(0, java_vector_utils.of(java_array))
        return np.array([res.get(i) for i in range(res.size())])


class PreprocessingTrainer(UnsupervisedTrainer):
    """Preprocessing trainer.
    """
    def __init__(self, proxy):
        """Constructs a new instance of PreprocessingTrainer.
        """
        Proxy.__init__(self, proxy)

    def fit(self, X, preprocessing=None):
        X_java = gateway.new_array(gateway.jvm.double, len(X), len(X[0]))

        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] is not None:
                    X_java[i][j] = float(X[i][j])
                else:
                    X_java[i][j] = float('NaN')

        java_model = self.proxy.fit(X_java, preprocessing)

        return PreprocessingModel(java_model)

class MinMaxScalerTrainer(PreprocessingTrainer):
    """Min-max scaler trainer.
    """
    def __init__(self):
        """Constructs a new instance of min-max scaler trainer.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.minmaxscaling.MinMaxScalerTrainer()

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class MaxAbsScalerTrainer(PreprocessingTrainer):
    """Max absolute scaler trainer.
    """
    def __init__(self):
        """Constructs a new instance of max absolute scaler trainer.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.maxabsscaling.MaxAbsScalerTrainer()

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class BinarizationTrainer(PreprocessingTrainer):
    """Binarization trainer.
    """
    def __init__(self, threshold=0.0):
        """Constructs a new instance of binarization trainer.

        Parameters
        ----------
        threshold : Threshold (Default value is 0).
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.binarization.BinarizationTrainer()
        proxy.withThreshold(threshold)

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class EncoderTrainer(PreprocessingTrainer):
    """Encoder trainer.
    """
    def __init__(self, encoded_features=[], encoder_indexing_strategy='frequency_desc', encoder_type='one_hot'):
        """Constructs a new instance of encoder trainer.

        Parameters
        ----------
        encoder_features : Encoded features (Default value is []).
        encoder_indexing_strategy : Encoder indexing strategy ('frequency_desc', 'frequency_asc', default value is 'frequency_desc').
        encoder_type : Encoder type ('one_hot', 'string', default value is 'one_hot').
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.encoding.EncoderTrainer()
        if not encoded_features:
            encoded_features = []
        for encoded_feature in encoded_features:
            proxy.withEncodedFeature(encoded_feature)

        java_encoder_indexing_strategy = None
        if encoder_indexing_strategy == 'frequency_desc':
            java_encoder_indexing_strategy = gateway.jvm.org.apache.ignite.ml.preprocessing.encoding.EncoderSortingStrategy.FREQUENCY_DESC
        elif encoder_indexing_strategy == 'frequency_asc':
            java_encoder_indexing_strategy = gateway.jvm.org.apache.ignite.ml.preprocessing.encoding.EncoderSortingStrategy.FREQUENCY_ASC
        elif encoder_indexing_strategy:
            raise Exception('Unknown encoder indexing strategy: %s' % encoder_indexing_strategy)
        proxy.withEncoderIndexingStrategy(java_encoder_indexing_strategy)
        
        java_encoder_type = None
        if encoder_type == 'one_hot':
            java_encoder_type = gateway.jvm.org.apache.ignite.ml.preprocessing.encoding.EncoderType.ONE_HOT_ENCODER
        elif encoder_type == 'string':
            java_encoder_type = gateway.jvm.org.apache.ignite.ml.preprocessing.encoding.EncoderType.STRING_ENCODER
        elif encoder_type:
            raise Exception("Unknown encoder type: %s" % encoder_type)
        proxy.withEncoderType(java_encoder_type)

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class ImputerTrainer(PreprocessingTrainer):
    """Imputer trainer.
    """
    def __init__(self, imputing_strategy='mean'):
        """Constructs a new instance of imputer trainer.

        Parameters
        ----------
        imputing_strategy : Imputing strategy ('mean', 'most_frequent', default value is 'mean').
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.imputing.ImputerTrainer()

        java_imputing_strategy = None
        if imputing_strategy == 'mean':
            java_imputing_strategy = gateway.jvm.org.apache.ignite.ml.preprocessing.imputing.ImputingStrategy.MEAN
        elif imputing_strategy == 'most_frequent':
            java_imputing_strategy = gateway.jvm.org.apache.ignite.ml.preprocessing.imputing.ImputingStrategy.MOST_FREQUENT
        elif imputing_strategy:
            raise Exception("Unknown imputing strategy: %s" % imputing_strategy)
        proxy.withImputingStrategy(java_imputing_strategy)

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class NormalizationTrainer(PreprocessingTrainer):
    """Normalization trainer.
    """
    def __init__(self, p=2):
        """Constructs a new instance of normalization trainer.

        Parameters
        ----------
        p : Degree of L space parameter value.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.normalization.NormalizationTrainer()
        proxy.withP(p)

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))

class StandardScalerTrainer(PreprocessingTrainer):
    """Standard scaler trainer.
    """
    def __init__(self):
        """Constructs a new instance of standard scaler trainer.
        """
        proxy = gateway.jvm.org.apache.ignite.ml.preprocessing.standardscaling.StandardScalerTrainer()

        PreprocessingTrainer.__init__(self, gateway.jvm.org.apache.ignite.ml.python.PythonPreprocessingTrainer(proxy))