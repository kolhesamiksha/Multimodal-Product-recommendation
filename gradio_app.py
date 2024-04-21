from pathlib import Path
from time import perf_counter
from typing import Optional
import torch
import gradio as gr
from PIL import Image

from logger import logger
from models.model_utils import ModalityType, get_embeddings, get_model, pinecone_retriever

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', help="Specify your pinecone index name")
parser.add_argument('-k', '--topk', help="no of images to recommend or retrieve")

arguments = parser.parse_args()
model = get_model()
logger.info("Model Loaded!")

pinecone_index_name = arguments.index
topk = int(arguments.topk)

def search_button_handler(
    text_query: Optional[str],
    image_query: Optional[Image.Image],
    audio_mic_query: Optional[str],
    audio_file_query: Optional[str],
    limit: int = 15,
):
    print(audio_file_query)
    if (
        not text_query
        and not image_query
        and not audio_mic_query
        and not audio_file_query
    ):
        logger.info("No inputs!")
        return
    # we have to pass a list for each query
    audio_query = None
    if text_query == "" and len(text_query) <= 0:
        text_query = None
    if text_query is not None:
        text_query = [text_query]
    if image_query is not None:
        image_query = [image_query]
    if audio_mic_query is not None:
        audio_query = [audio_mic_query]
    if audio_file_query is not None:
        audio_query = [audio_file_query]
    start = perf_counter()
    logger.info(f"Searching ...")
    embeddings = get_embeddings(model, text_query, image_query, audio_query).values()
    # if multiple inputs, we sum them
    embedding = torch.vstack(list(embeddings)).sum(0)
    logger.info(
        f"Model took {(perf_counter() - start) * 1000:.2f}"
    )
    query_res = pinecone_retriever(embedding.numpy(), pinecone_index_name, topk)
    image_urls = [match['metadata']['image_url'] for match in query_res['matches']]
    return image_urls


def clear_button_handler():
    return [None] * 6


css = """
#image_query .output-image, .input-image, .image-preview { height: 100px !important; }
#gallery {
    display: grid;
    grid-template-columns: repeat(4, minmax(500px, 1fr)); /* Display four columns */
    grid-gap: 20px; /* Adjust the gap between grid items */
}

#gallery img {
    max-width: 100%;
    height: auto; /* Let the height adjust automatically to maintain aspect ratio */
}
#audio_file_query { height: 100px; }
"""
with gr.Blocks(css=css) as demo:
    # pairs of (input_type, data, +/-)
    inputs = gr.State([])
    with Path("docs/APP_README.md").open() as f:
        gr.Markdown(f.read())
    text_query = gr.Text(label="Text")
    with gr.Row():
        image_query = gr.Image(label="Image", type="pil", elem_id="image_query")

        with gr.Column():
            audio_mic_query = gr.Audio(
                label="Audio", sources="microphone", type="filepath"
            )
            audio_file_query = gr.Audio(
                label="Audio", type="filepath", elem_id="audio_file_query"
            )
    markdown = gr.Markdown("")
    search_button = gr.Button("Search", variant="primary")
    clear_button = gr.Button("Clear", variant="secondary")
    with gr.Accordion("Settings", open=False):
        limit = gr.Slider(
            minimum=1,
            maximum=30,
            value=15,
            step=1,
            label="search limit",
            interactive=True,
        )
    gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery")
    clear_button.click(
        clear_button_handler,
        [],
        [text_query, image_query, audio_mic_query, audio_file_query, markdown, gallery],
    )
    search_button.click(
        search_button_handler,
        [text_query, image_query, audio_mic_query, audio_file_query, limit],
        [gallery],
    )

demo.queue()
demo.launch(server_name="0.0.0.0", share=True, server_port=7860)
