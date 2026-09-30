"""Скрипт для программной загрузки ZIP в Qdrant (без Streamlit UI)"""
import os
import sys
import uuid
import zipfile
import tempfile
import shutil
import requests
import pandas as pd
from pathlib import Path

import re
import time

EMBEDDING_SERVICE_URL = "http://localhost:5000"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = sys.argv[2] if len(sys.argv) > 2 else "sadykhan_config"
ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else None
ROW_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 20
MAX_RETRIES = 5

def camel_words(name):
    """ЧекККМ -> Чек ККМ: модели лучше понимают имя 1С, разбитое на слова."""
    s = re.sub(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", " ", name)
    s = re.sub(r"(?<=[A-ZА-ЯЁ])(?=[A-ZА-ЯЁ][a-zа-яё])", " ", s)
    return s.replace("_", " ")

def get_embedding_info():
    r = requests.get(f"{EMBEDDING_SERVICE_URL}/model-info", timeout=30)
    r.raise_for_status()
    return r.json()

def generate_embeddings(texts, retry=0):
    try:
        r = requests.post(f"{EMBEDDING_SERVICE_URL}/embed",
                          json={"texts": texts, "task": "retrieval.passage"},
                          timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        if retry < MAX_RETRIES:
            wait = 2 ** retry
            print(f"  Retry {retry+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
            return generate_embeddings(texts, retry + 1)
        raise

def generate_embeddings_batched(texts):
    all_emb = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i+EMBEDDING_BATCH_SIZE]
        emb = generate_embeddings(batch)
        all_emb.extend(emb)
        print(f"  Embeddings: {min(i+EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_emb

def main():
    if not ZIP_PATH:
        print("Использование: python upload_zip.py <path_to_zip> [collection_name]")
        sys.exit(1)

    print(f"ZIP: {ZIP_PATH}")
    print(f"Коллекция: {COLLECTION_NAME}")

    # Проверка сервисов
    print("\n1. Проверка сервисов...")
    info = get_embedding_info()
    dimensions = info.get("dimensions", 384)
    print(f"   Embedding: {info.get('model_name')}, dim={dimensions}")

    r = requests.get(f"{QDRANT_URL}/collections")
    r.raise_for_status()
    print(f"   Qdrant: OK")

    # Извлечение ZIP
    print("\n2. Извлечение ZIP...")
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(temp_dir)

        # Поиск objects.csv
        csv_path = None
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if f.lower() == 'objects.csv':
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break

        if not csv_path:
            print("ОШИБКА: objects.csv не найден в архиве!")
            sys.exit(1)

        base_path = os.path.dirname(csv_path)

        # Чтение CSV
        # keep_default_na=False: пустой синоним остаётся пустой строкой, а не 'nan'
        df = pd.read_csv(csv_path, encoding='utf-8', sep=';', quotechar='"', dtype=str, keep_default_na=False)
        print(f"   Найдено {len(df)} объектов")

        required = ["Имя объекта", "Тип объекта", "Синоним", "Файл"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ОШИБКА: отсутствуют колонки: {missing}")
            sys.exit(1)

        # Создание/пересоздание коллекции
        print(f"\n3. Создание коллекции '{COLLECTION_NAME}'...")
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if r.status_code == 200:
            print(f"   Удаление существующей коллекции...")
            requests.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")

        requests.put(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", json={
            "vectors": {
                "object_name": {"size": dimensions, "distance": "Cosine", "on_disk": True},
                "friendly_name": {"size": dimensions, "distance": "Cosine", "on_disk": True}
            }
        }).raise_for_status()
        print(f"   Коллекция создана (dim={dimensions})")

        # Обработка батчами
        print(f"\n4. Загрузка данных...")
        total = len(df)
        total_loaded = 0

        for i in range(0, total, ROW_BATCH_SIZE):
            batch_df = df.iloc[i:i+ROW_BATCH_SIZE]
            obj_texts = []
            friendly_texts = []
            metadatas = []

            for _, row in batch_df.iterrows():
                obj_name = row["Имя объекта"]
                obj_type = row["Тип объекта"]
                synonym = row["Синоним"]
                file_name = row["Файл"]

                # Читаем MD файл
                md_path = Path(base_path) / file_name
                doc = ""
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        doc = f.read()
                except Exception as e:
                    print(f"   Предупреждение: не удалось прочитать {file_name}: {e}")

                friendly = f"{obj_type}: {synonym}" if synonym else obj_type
                obj_texts.append(f"{obj_name} {camel_words(obj_name)}")
                friendly_texts.append(friendly)
                metadatas.append({
                    "object_name": obj_name,
                    "object_type": obj_type,
                    "doc": doc,
                    "file_name": file_name,
                    "friendly_name": friendly
                })

            if not obj_texts:
                continue

            print(f"\n   Батч {i//ROW_BATCH_SIZE + 1}: строки {i+1}-{min(i+ROW_BATCH_SIZE, total)}")

            # Генерация эмбеддингов
            print("   Генерация object_name эмбеддингов...")
            obj_embeddings = generate_embeddings_batched(obj_texts)

            print("   Генерация friendly_name эмбеддингов...")
            friendly_embeddings = generate_embeddings_batched(friendly_texts)

            # Загрузка в Qdrant
            points = []
            for j in range(len(obj_texts)):
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": {
                        "object_name": obj_embeddings[j],
                        "friendly_name": friendly_embeddings[j]
                    },
                    "payload": metadatas[j]
                })

            r = requests.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                json={"points": points}
            )
            r.raise_for_status()
            total_loaded += len(points)
            print(f"   Загружено: {total_loaded}/{total}")

        print(f"\nGotovo! Zagrujeno {total_loaded} obektov v kollekciju '{COLLECTION_NAME}'")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
