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

from functools import lru_cache

import sherpa_onnx
from huggingface_hub import hf_hub_download

sample_rate = 16000


def _get_nn_model_filename(
    repo_id: str,
    filename: str,
    subfolder: str = "exp",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def _get_bpe_model_filename(
    repo_id: str,
    filename: str = "bpe.model",
    subfolder: str = "data/lang_bpe_500",
) -> str:
    bpe_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return bpe_model_filename


def _get_token_filename(
    repo_id: str,
    filename: str = "tokens.txt",
    subfolder: str = "data/lang_char",
) -> str:
    token_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return token_filename


@lru_cache(maxsize=10)
def _get_whisper_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    name = repo_id.split("-")[1]
    assert name in ("tiny.en", "base.en", "small.en", "medium.en"), repo_id
    full_repo_id = "csukuangfj/sherpa-onnx-whisper-" + name
    encoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-encoder.int8.ort",
        subfolder=".",
    )

    decoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-decoder.int8.ort",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=full_repo_id, subfolder=".", filename=f"{name}-tokens.txt"
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_paraformer_zh_pre_trained_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


@lru_cache(maxsize=10)
def _get_russian_pre_trained_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "alphacep/vosk-model-ru",
        "alphacep/vosk-model-small-ru",
    ), repo_id

    if repo_id == "alphacep/vosk-model-ru":
        model_dir = "am-onnx"
    elif repo_id == "alphacep/vosk-model-small-ru":
        model_dir = "am"

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder.onnx",
        subfolder=model_dir,
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder.onnx",
        subfolder=model_dir,
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner.onnx",
        subfolder=model_dir,
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="lang")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    return recognizer


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
