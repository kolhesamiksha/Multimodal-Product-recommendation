from typing import List, Union, Optional, Dict

import torch
import numpy as np
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
import pickle
import os
from .data import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data
from .imagebind_model import ModalityType, imagebind_huge

import hashlib
from Crypto.Cipher import AES 
from Crypto.Util.Padding import pad, unpad 

from dotenv import load_dotenv

load_dotenv(override=True)  #override=True, always consider API_key from .env file

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device= "cpu"
ImageLike = Union['str', np.ndarray, Image.Image]

def encrypt_pass(key):
    random_input_text = b"Xysjgkyur8k4l8p5"
    cipher = AES.new(random_input_text, AES.MODE_CBC)
    key_bytes = key.encode('utf-8')
    cipher_text = cipher.encrypt(pad(key_bytes, AES.block_size))

    return cipher_text, cipher.iv
# secure your API key with cryptography
def decrypt_pass(encrypted_key, iv):
    random_input_text = b"Xysjgkyur8k4l8p5"
    iv = iv[:16]
    cipher = AES.new(random_input_text, AES.MODE_CBC, iv)
    plain_text = unpad(cipher.decrypt(encrypted_key), AES.block_size)
    return plain_text.decode('utf-8')


def get_model(dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    model = imagebind_huge(pretrained=True)
    model = model.eval().to(device, dtype=dtype)
    return model


@torch.no_grad()
def get_texts_embeddings(model: torch.nn.Module, texts: List[str]) -> torch.Tensor:
    inputs = {ModalityType.TEXT: load_and_transform_text(texts, device)}
    texts_embeddings = model(inputs)[ModalityType.TEXT]
    return texts_embeddings


@torch.no_grad()
def get_images_embeddings(
    model: torch.nn.Module, images: List[ImageLike],dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    inputs = {ModalityType.VISION: load_and_transform_vision_data(images, device).to(dtype)}
    images_embeddings = model(inputs)[ModalityType.VISION]
    return images_embeddings


@torch.no_grad()
def get_embeddings(
    model: torch.nn.Module,
    texts: Optional[List[str]],
    images: Optional[List[ImageLike]],
    audio: Optional[List[str]],
    dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:  
    inputs = {}
    if texts is not None:
        # they need to be ints
        inputs[ModalityType.TEXT] = load_and_transform_text(texts, device)
    if images is not None:
        inputs[ModalityType.VISION] = load_and_transform_vision_data(images, device)
    if audio is not None:
        inputs[ModalityType.AUDIO] = load_and_transform_audio_data(audio, device)
    embeddings = model(inputs)
    return embeddings

def pinecone_retriever(query_embedding, pinecone_index_name, topk):
    encrypted_key, iv = encrypt_pass(os.getenv("PINECONE_API_KEY"))
    pinecone = Pinecone(
        api_key = decrypt_pass(encrypted_key, iv)
    )

    my_index_name = pinecone_index_name
    my_index = pinecone.Index(name = my_index_name)

    result = my_index.query(vector=query_embedding.tolist(), top_k=topk, include_metadata=True)
    return result