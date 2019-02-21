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

class SupervisedTrainer:
    """Supervised trainer.
    """
    @abstractmethod
    def fit(self, data, feature_extractor, label_extractor):
        """Trains model based on data.

        Parameters
        ----------

        data : Apache Ignite cache.

        feature_extractor : Feature extractor.

        label_extractor : Label extractor.
        """
        pass

    @abstractmethod
    def update(self, mdl, data, feature_extractor, label_extractor):
        """Updates the model.

        Parameters
        ----------

        mdl : Model.

        data : Apache Ignite cache.

        feature_extractor : Feature extractor.

        label_extractor : Label extractor.
        """
        pass

    @abstractmethod
    def is_updatable(mdl)
        """Checks if model is updatable.

        Parameters
        ----------

        mdl : Model.
        """
        pass

class UnsupervisedTrainer:
    """Unsupervised trainer.
    """
    @abstractmethod
    def fit(self, data, feature_extractor):
        """Trains model based on data.

        Parameters
        ----------

        data : Apache Ignite cache.

        feature_extractor : Feature extractor.
        """
        pass

    @abstractmethod
    def update(self, mdl, data, feature_extractor):
        """Updates the model.

        Parameters
        ----------

        mdl : Model.

        data : Apache Ignite cache.

        feature_extractor : Feature extractor.
        """
        pass

    @abstractmethod
    def is_updatable(mdl)
        """Checks if model is updatable.

        Parameters
        ----------

        mdl : Model.
        """
        pass
