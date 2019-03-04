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

import unittest

from ignite_ml.regression import LinearRegressionTrainer

class TestRegressions(unittest.TestCase):

    def test_linear_regression(self):
        trainer = LinearRegressionTrainer()
        model = trainer.fit([[1.0], [2.0]], [2.0, 4.0])
        prediction = model.predict([3.0])
        self.assertAlmostEqual(6, prediction, delta=1e-5)

    def test_linear_regression_array_prediction(self):
        trainer = LinearRegressionTrainer()
        model = trainer.fit([[1.0], [2.0]], [2.0, 4.0])
        prediction = model.predict([[0], [1], [2]])
        for i in [0, 1, 2]:
            self.assertAlmostEqual(i * 2, prediction[i], delta=1e-5)

    def test_decision_tree_regression(self):
        pass

    def test_decision_tree_regression_array_prediction(self):
        pass

    def test_knn_regression(self):
        pass

    def test_knn_regression_array_prediction(self):
        pass

    def test_random_forest_regression(self):
        pass

    def test_random_forest_regression_array_prediction(self):
        pass
