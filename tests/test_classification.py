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

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from ignite_ml.classification import DecisionTreeClassificationTrainer
from ignite_ml.classification import ANNClassificationTrainer
from ignite_ml.classification import KNNClassificationTrainer
from ignite_ml.classification import LogRegClassificationTrainer
from ignite_ml.classification import SVMClassificationTrainer
from ignite_ml.classification import RandomForestClassificationTrainer
from ignite_ml.common import MLPArchitecture
from ignite_ml.classification import MLPClassificationTrainer

class TestClassification(unittest.TestCase):

    def test_decision_tree_classification(self):
        ignite_trainer = DecisionTreeClassificationTrainer()
        sklearn_trainer = DecisionTreeClassifier()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_ann_classification(self):
        ignite_trainer = ANNClassificationTrainer()
        sklearn_trainer = ANNClassificationTrainer()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_knn_classification(self):
        ignite_trainer = KNNClassificationTrainer()
        sklearn_trainer = KNeighborsClassifier()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_log_reg_classification(self):
        ignite_trainer = LogRegClassificationTrainer()
        sklearn_trainer = LogisticRegression()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_svm_classification(self):
        ignite_trainer = SVMClassificationTrainer()
        sklearn_trainer = LinearSVC()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

    def test_random_forest_classification(self):
        ignite_trainer = RandomForestClassificationTrainer()
        sklearn_trainer = RandomForestClassifier()
        self.__assert_trainer_single_prediction(ignite_trainer, sklearn_trainer)

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
        x, y = make_classification(random_state=42, n_features=20, n_informative=10, n_samples=100)
        return (x, x, y, y)

if __name__ == '__main__':
    unittest.main()
