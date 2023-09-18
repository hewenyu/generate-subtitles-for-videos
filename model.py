# Copyright      2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
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

from huggingface_hub import hf_hub_download

english_models = {
    "whisper-tiny.en": _get_whisper_model,
    "whisper-base.en": _get_whisper_model,
    "whisper-small.en": _get_whisper_model,
}

chinese_english_mixed_models = {
    "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28": _get_paraformer_zh_pre_trained_model,
}

russian_models = {
    "alphacep/vosk-model-ru": _get_russian_pre_trained_model,
    "alphacep/vosk-model-small-ru": _get_russian_pre_trained_model,
}

language_to_models = {
    "English": list(english_models.keys()),
    "Chinese+English": list(chinese_english_mixed_models.keys()),
    "Russian": list(russian_models.keys()),
}
