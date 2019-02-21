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

from ..common import SupervisedTrainer

class MLP(SupervisedTrainer):
    """ Multilayer perceptron trainer.
    """
    def __init__(self, env_builder=None, label_converter=None, arch, loss,
                 update_strategy, max_iter, batch_size, max_loc_iter, seed):
        """ Multilayer perceptron trainer.

        Parameters
        ----------
        env_builder : Environment builder.
        label_converter : Label converter.
        arch : Architecture.
        loss : Loss.
        update_strategy : Update strategy.
        max_iter : Max number of iterations.
        batch_size : Batch size.
        max_loc_iter : Max number of local iterations.
        seed : Seed.
        """
        pass
