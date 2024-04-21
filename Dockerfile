FROM python:3.8

# Set the WORKDIR: where to copy the entire files
WORKDIR /usr/src/app

#Imagebind model~4GB
COPY .checkpoints/ .checkpoints/  
COPY audio audio
COPY images images
COPY models models
COPY docs docs

COPY gradio_app.py .
COPY requirements.txt .
COPY .env .

RUN pip install -r requirements.txt

CMD ["python", "gradio_app.py", "-i", "pinterest-multimodal-search", "-k", "8"]




