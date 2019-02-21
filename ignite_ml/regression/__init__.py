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

from ..common import SupervisedTrainer

class Regressor(SupervisedTrainer):
    """Regression.
    """
    pass

class DecisionTreeRegressor(Regressor):
    """Decision Tree Regressor.
    """
    def __init__(self, env_builder=None, label_converter=None, max_deep=None,
                 min_impurity_decrease=None, compressor=None, use_index=True):
        """Decision Tree Regressor.

        Parameters
        ----------

        env_builder : Environment builder.

        label_converter : Label converter.
        """
        pass

class KNNRegressor(Regressor):
    """KNN Regressor.
    """
    def __init__(self, env_builder=None, label_converter=None):
        """KNN Regressor.

        Parameters
        ----------

        env_builder : Environment builder.

        label_converter : Label converter.
        """
        pass

class LinearRegressor(Regressor):
    """Linear Regressor.
    """
    def __init__(self, env_builder=None, label_converter=None):
        """Linear Regressor.

        Parameters
        ----------

        env_builder : Environment builder.

        label_converter : Label converter.
        """
        pass

class RandomForestRegressor(Regressor):
    """Random Forest Regressor.
    """
    def __init__(self, env_builder=None, label_converter=None,
                  trees=None, sub_sample_size=None, max_depth=None,
                  min_impurity_delta=None, features_count_selection_strategy=None,
                  nodes_to_learn_selection_strategy=None, seed=None):
        """Random Forest Regressor

        Parameters
        ----------

        env_builder : Environment builder.

        label_converter : Label converter.
        """
        pass
