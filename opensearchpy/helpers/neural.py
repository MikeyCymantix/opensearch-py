# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
#
#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import collections.abc as collections_abc
from typing import Any, Optional

from .utils import DslBase

import collections.abc as collections_abc
from typing import Any, Dict


def SS(name: Any, **params: Any):
    """
    Shortcut function to create NeuralSearch objects.

    :param name: Can be a string indicating the name of the search class, or a mapping containing parameters.
    :param params: Additional parameters passed to the search class.
    :return: An instance of a NeuralSearch subclass.
    :raises ValueError: If improper input is provided.
    """
    if isinstance(name, collections_abc.Mapping):
        if params:
            raise ValueError("SS() cannot accept parameters when passing a dict.")

        ss: Dict = name.copy()  # Make a copy to avoid modifying the original dictionary

        if "neural_query" in ss:
            neural_query_params = ss.pop("neural_query")
        else:
            neural_query_params = {}

        if len(ss) == 1:
            embedding_field, field_params = ss.popitem()
            neural_query_params["embedding_field"] = embedding_field
            neural_query_params.update(field_params)
        else:
            raise ValueError(
                "Expected a single embedding field key, but got multiple or none."
            )

        return NeuralSearch.get_dsl_class("neural_query")(**neural_query_params)

    elif isinstance(name, str):
        return NeuralSearch.get_dsl_class(name)(**params)

    else:
        raise ValueError(f"Invalid name type for SS: {type(name)}")


class NeuralSearch(DslBase):
    _type_name = "neural_query"
    _type_shortcut = staticmethod(SS)
    name: Optional[str] = None
    _param_defs = {
        "embedding_field": {"type": "field"},
        "should": {"type": "field"},
        "must_not": {"type": "field"},
        "filter": {"type": "field"},
    }


class EmbeddingField(NeuralSearch):
    name = "embedding_field"


class ModelId(NeuralSearch):
    name = "model_id"


class QueryText(NeuralSearch):
    name = "query_text"


class K(NeuralSearch):
    name = "k"

class QueryImage(NeuralSearch):
    name = "query_image"
