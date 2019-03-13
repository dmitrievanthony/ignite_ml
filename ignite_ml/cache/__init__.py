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

"""Classification trainers.
"""

import numpy as np

from ..common import Proxy
from ..common import Utils

from ..common import gateway

class Ignition:

    __ignite = None

    def ignite(cfg):
        if Ignition.__ignite is None:
            java_ignite = gateway.jvm.org.apache.ignite.Ignition.start(cfg)
            Ignition.__ignite = Ignite(java_ignite)
        return Ignition.__ignite

class Ignite(Proxy):

    def __init__(self, proxy):
        Proxy.__init__(self, proxy)
    
    def getCache(self, name):
        java_cache = self.proxy.cache(name)
        return Cache(java_cache)

    def createCache(self, name, parts=10):
        affinity = gateway.jvm.org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction(False, parts)
        cc = gateway.jvm.org.apache.ignite.configuration.CacheConfiguration()
        cc.setName(name)
        cc.setAffinity(affinity)
        java_cache = self.proxy.createCache(cc)
        return Cache(java_cache)

class Cache(Proxy):

    def __init__(self, proxy):
        Proxy.__init__(self, proxy)

    def get(self, key):
        java_array = self.proxy.get(key)
        return Utils.from_java_double_array(java_array)

    def put(self, key, value):
        value = Utils.to_java_double_array(value)
        self.proxy.put(key, value)

    def getAll(self):
        raise Exception("Not implemented yet")

    def putAll(self, keys, values):
        raise Exception("Not implemented yet")

    def size(self):
        return self.proxy.size()
