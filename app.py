#!/usr/bin/env python3
#
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

# References:
# https://gradio.app/docs/#dropdown

import shutil
import logging
import os
from pathlib import Path

import gradio as gr

from decode import decode
from model import get_pretrained_model, get_vad, language_to_models, get_file

title = "# Next-gen Kaldi: Generate subtitles for videos"

description = """
This space shows how to generate subtitles/captions with Next-gen Kaldi.

It is running on CPU within a docker container provided by Hugging Face.

See more information by visiting the following links:

- <https://github.com/k2-fsa/sherpa-onnx>
- <https://github.com/k2-fsa/icefall>
- <https://github.com/k2-fsa/k2>
- <https://github.com/lhotse-speech/lhotse>

If you want to deploy it locally, please see
<https://k2-fsa.github.io/sherpa/>
"""

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""

examples = [
    [
        "English",
        "whisper-tiny.en",
        "President-Obama-on-the-Importance-of-Education.mp4",
    ],
    ["English", "whisper-tiny.en", "jobs-at-stanford.mp4"],
]

for _, _, name in examples:
    filename = get_file(
        "csukuangfj/vad",
        name,
        subfolder=".",
    )

    shutil.copyfile(filename, name)


def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown.update(choices=choices, value=choices[0])

    raise ValueError(f"Unsupported language: {language}")


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def show_file_info(in_filename: str):
    logging.info(f"Input file: {in_filename}")
    _ = os.system(f"ffprob -hide_banner -i '{in_filename}'")


def process_uploaded_file(
    language: str,
    repo_id: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first upload a file and then click "
            'the button "submit for recognition"',
            "result_item_error",
        )

    logging.info(f"Processing uploaded file: {in_filename}")

    recognizer = get_pretrained_model(repo_id)
    vad = get_vad()

    result = decode(recognizer, vad, in_filename)

    srt_filename = Path(in_filename).with_suffix(".srt")
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(result)

    return (
        (in_filename, srt_filename),
        srt_filename,
        build_html_output("Done! Please download the SRT file", "result_item_success"),
        result,
    )


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)
    language_choices = list(language_to_models.keys())

    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )

    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )

    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    with gr.Tabs():
        with gr.TabItem("Upload video from disk"):
            uploaded_file = gr.Video(
                source="upload",
                interactive=True,
                label="Upload from disk",
                show_share_button=True,
            )
            upload_button = gr.Button("Submit for recognition")

            output_video = gr.Video(label="Output")
            output_srt_file = gr.File(label="Generated subtitles", show_label=True)

            output_info = gr.HTML(label="Info")
            output_textbox = gr.Textbox(label="Recognized speech from uploaded file")

            gr.Examples(
                examples=examples,
                inputs=[
                    language_radio,
                    model_dropdown,
                    uploaded_file,
                ],
                outputs=[
                    output_video,
                    output_srt_file,
                    output_info,
                    output_textbox,
                ],
                fn=process_uploaded_file,
            )

        upload_button.click(
            process_uploaded_file,
            inputs=[
                language_radio,
                model_dropdown,
                uploaded_file,
            ],
            outputs=[
                output_video,
                output_srt_file,
                output_info,
                output_textbox,
            ],
        )

    gr.Markdown(description)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    demo.launch()
