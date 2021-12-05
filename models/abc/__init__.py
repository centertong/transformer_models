# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from transformers.file_utils import is_torch_available #_LazyModule, is_tokenizers_available, is_torch_available
from transformers import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    import tokenizers
    _tokenizers_available = True  # pylint: disable=invalid-name
    logger.info("Tokenizers version {} available.".format(tokenizers.__version__))
except ImportError:
    _tokenizers_available = False  # pylint: disable=invalid-name


def is_tokenizers_available():
    return _tokenizers_available
    




if is_torch_available():
    from .modeling_abc import (
        AbcForMaskedLM,
        AbcLayer,
        AbcModel,
        AbcPreTrainedModel,
        AbcForSequenceClassification,
    )