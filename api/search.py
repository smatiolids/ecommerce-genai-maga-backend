from app import session, keyspace
import cassio
import os
from flask import Blueprint, request, jsonify
import json

import torch
import clip
from PIL import Image
import random
import string

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)


# Init storage
v_store = cassio.table.MetadataVectorCassandraTable(session=session,
                                                    keyspace=keyspace,
                                                    table="demo_ecommerce",
                                                    vector_dimension=512)


bp = Blueprint('search', __name__)


# @bp.route('/image-embedding', methods=['POST'])
# def get_image_embedding():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']

#     allowed_extensions = {'png', 'jpg', 'jpeg'}
#     if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
#         return jsonify({'error': 'Invalid file extension'}), 400

#     file_path = f'tmp/temp_image_{generate_random_string()}.jpg'
#     file.save(file_path)

#     embedding = embed_image(file_path)
#     os.remove(file_path)
#     print(embedding)

#     return jsonify({'embedding': embedding}), 200


@bp.route('/search-by-image', methods=['POST'])
def search_by_image():
    print("request")
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension'}), 400

    file_path = f'tmp/temp_image_{generate_random_string()}.jpg'
    file.save(file_path)

    embedding = embed_image(file_path)

    os.remove(file_path)

    results = v_store.metric_ann_search(n=10, vector=embedding, metric='dot')
    data = []
    for row in results:
        data.append({'title': row['body_blob'],
                    'file': row['metadata']['filename'],
                     'distance': row['distance']})


@bp.route('/search', methods=['POST'])
def search_combined():
    file_path = None

    if 'file' in request.files:
        file = request.files['file']
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file extension'}), 400
        file_path = f'tmp/temp_image_{generate_random_string()}.jpg'
        file.save(file_path)

    post = json.loads(request.form['post'])

    query = None
    n = 20
    if 'query' in post:
        query = post['query']

    if 'n' in post:
        n = post['n']

    embedding = None
    if file_path is not None and query is not None:
        # Multimodal
        embedding = get_clip_embedding(post['query'], file_path)
    elif file_path is None and query is not None:
        # Text Search
        embedding = embed_query(query)
    elif file_path is not None and query is None:
        # Image Search
        embedding = embed_image(file_path)

    if file_path is not None:
        os.remove(file_path)

    if embedding is None:
        return jsonify({'error': 'Invalid file extension'}), 400
    else:
        results = v_store.metric_ann_search(
            n=n, vector=embedding, metric='dot')
        data = []
        for row in results:
            data.append({'title': row['body_blob'],
                         'file': row['metadata']['filename'],
                         'distance': row['distance']})

        return jsonify({'data': data}), 200


@bp.route('/text-embedding', methods=['POST'])
def get_text_embedding():
    if not request.is_json:
        return jsonify({'error': 'Request body must be in JSON format'}), 400
    data = request.json
    query = data['query']
    embedding = embed_query(query)

    return jsonify({'embedding': embedding}), 200


@bp.route('/search-by-text', methods=['POST'])
def search_by_text_embedding():
    if not request.is_json:
        return jsonify({'error': 'Request body must be in JSON format'}), 400
    data = request.json
    query = data['query']
    embedding = embed_query(query)
    n = 20
    if 'n' in data:
        n = data['n']

    results = v_store.metric_ann_search(n=n, vector=embedding, metric='dot')
    data = []
    for row in results:
        data.append({'title': row['body_blob'],
                    'file': row['metadata']['filename'],
                     'distance': row['distance']})
    # print(results)

    return jsonify({'data': data}), 200


def generate_random_string(length=15):
    # You can customize this further if needed
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def embed_query(q):
    query_embed = clip.tokenize(q, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(query_embed)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.numpy().tolist()[0]


def embed_image(image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.numpy().tolist()[0]


def get_clip_embedding(text, image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(text, truncate=True).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    averaged_features = (image_features + text_features) / 2
    return averaged_features.numpy().tolist()[0]
