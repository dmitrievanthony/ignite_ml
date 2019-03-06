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

from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

from ignite_ml.regression import LinearRegressionTrainer
from ignite_ml.regression import DecisionTreeRegressionTrainer
from ignite_ml.regression import KNNRegressionTrainer
from ignite_ml.regression import RandomForestRegressionTrainer
from ignite_ml.common import MLPArchitecture
from ignite_ml.regression import MLPRegressionTrainer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

class TestRegressions(unittest.TestCase):

    def test_linear_regression_single_prediction(self):
        ignite_trainer = LinearRegressionTrainer()
        sklearn_trainer = LinearRegression()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_linear_regression_array_prediction(self):
        ignite_trainer = LinearRegressionTrainer()
        sklearn_trainer = LinearRegression()
        self.__assert_trainer_array_prediction(ignite_trainer, sklearn_trainer)

    def test_decision_tree_regression_single_prediction(self):
        ignite_trainer = DecisionTreeRegressionTrainer(max_deep=100)
        sklearn_trainer = DecisionTreeRegressor(max_depth=100)
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_decision_tree_regression_array_prediction(self):
        ignite_trainer = DecisionTreeRegressionTrainer(max_deep=100)
        sklearn_trainer = DecisionTreeRegressor(max_depth=100)
        self.__assert_trainer_array_prediction(ignite_trainer, sklearn_trainer)

    def test_knn_regression(self):
        ignite_trainer = KNNRegressionTrainer()
        sklearn_trainer = KNeighborsRegressor()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_knn_regression_array_prediction(self):
        ignite_trainer = KNNRegressionTrainer()
        sklearn_trainer = KNeighborsRegressor()
        self.__assert_trainer_array_prediction(ignite_trainer, sklearn_trainer)

    def test_random_forest_regression(self):
        ignite_trainer = RandomForestRegressionTrainer(trees=1, max_depth=100)
        sklearn_trainer = RandomForestRegressor(n_estimators=1, max_depth=100)
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        ignite_model = ignite_trainer.fit(x_train, y_train)
        sklearn_model = sklearn_trainer.fit(x_train, y_train)
        self.assertAlmostEqual(
            r2_score(y_test, sklearn_model.predict(x_test)),
            r2_score(y_test, ignite_model.predict(x_test)),
            delta=1e-5
        )

    def test_mlp_regression(self):
        x_train, x_test, y_train, y_test = self.__generate_dataset()
        trainer = MLPRegressionTrainer(MLPArchitecture(20).with_layer(1, activator='linear'))
        model = trainer.fit(x_train, y_train)
        self.assertTrue(r2_score(y_test, model.predict(x_test)) > 0.7)

    def __assert_trainer_single_prediction(self, ignite_trainer, sklearn_trainer):
        x_train, x_test, y_train, y_test = self.__generate_dataset()

        ignite_model = ignite_trainer.fit(x_train, y_train)
        sklearn_model = sklearn_trainer.fit(x_train, y_train)

        for i in range(len(x_test)):
            ignite_prediction = ignite_model.predict([x_test[i]])
            sklearn_prediction = sklearn_model.predict([x_test[i]])
            self.assertAlmostEqual(sklearn_prediction, ignite_prediction, delta=1e-5)

    def __assert_trainer_array_prediction(self, ignite_trainer, sklearn_trainer):
        x_train, x_test, y_train, y_test = self.__generate_dataset()

        ignite_model = ignite_trainer.fit(x_train, y_train)
        sklearn_model = sklearn_trainer.fit(x_train, y_train)

        ignite_prediction = ignite_model.predict(x_test)
        sklearn_prediction = sklearn_model.predict(x_test)

        for i in range(len(x_test)):
            self.assertAlmostEqual(sklearn_prediction[i], ignite_prediction[i], delta=1e-5) 

    def __generate_dataset(self):
        x, y = make_regression(random_state=42, n_features=20, n_informative=10, n_samples=100)
        return (x, x, y, y)

if __name__ == '__main__':
    unittest.main()
