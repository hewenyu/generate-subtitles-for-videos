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

import logging
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import sherpa_onnx

from model import sample_rate


@dataclass
class Segment:
    start: float
    duration: float
    text: str = ""

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        s = f"0{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"0{timedelta(seconds=self.end)}"[:-3]
        s = s.replace(".", ",")
        s += "\n"
        s += self.text
        return s


def decode(
    recognizer: sherpa_onnx.OfflineRecognizer,
    vad: sherpa_onnx.VoiceActivityDetector,
    punct: Optional[sherpa_onnx.OfflinePunctuation],
    filename: str,
) -> str:
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        filename,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frames_per_read = int(sample_rate * 100)  # 100 second

    window_size = 512

    buffer = []

    segment_list = []

    logging.info("Started!")

    all_text = []

    is_last = False

    while True:
        # *2 because int16_t has two bytes
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            if is_last:
                break
            is_last = True
            data = np.zeros(sample_rate, dtype=np.int16)

        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768

        buffer = np.concatenate([buffer, samples])
        while len(buffer) > window_size:
            vad.accept_waveform(buffer[:window_size])
            buffer = buffer[window_size:]

        streams = []
        segments = []
        while not vad.empty():
            segment = Segment(
                start=vad.front.start / sample_rate,
                duration=len(vad.front.samples) / sample_rate,
            )
            segments.append(segment)

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, vad.front.samples)

            streams.append(stream)

            vad.pop()

        for s in streams:
            recognizer.decode_stream(s)

        for seg, stream in zip(segments, streams):
            seg.text = stream.result.text.strip()
            if len(seg.text) == 0:
                logging.info("Skip empty segment")
                continue

            if len(all_text) == 0:
                all_text.append(seg.text)
            elif len(all_text[-1][0].encode()) == 1 and len(seg.text[0].encode()) == 1:
                all_text.append(" ")
                all_text.append(seg.text)
            else:
                all_text.append(seg.text)

            if punct is not None:
                seg.text = punct.add_punctuation(seg.text)
            segment_list.append(seg)
    all_text = "".join(all_text)
    if punct is not None:
        all_text = punct.add_punctuation(all_text)

    return "\n\n".join(f"{i}\n{seg}" for i, seg in enumerate(segment_list, 1)), all_text
