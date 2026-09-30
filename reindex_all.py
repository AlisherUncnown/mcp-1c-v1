"""
Полная переиндексация всех коллекций RAG текущей моделью embedding-service.

  python reindex_all.py            # все коллекции из SOURCES + REEMBED_ONLY
  python reindex_all.py zup erp_config

Перед пересозданием коллекции сверяет состав: если в новой выгрузке нашлось меньше
MIN_OVERLAP старых имён объектов - коллекция не трогается (источник, видимо, не тот).
Бэкап коллекций до переиндексации: qdrant_backup_<дата>/<коллекция>.snapshot.
"""
import sys
import time
import uuid
import json
import subprocess
import requests

import index_from_files as ix
from upload_zip import camel_words

QDRANT = "http://localhost:6333"
EMBED = "http://localhost:5000"
MIN_OVERLAP = 0.9

# Бережём ноутбучную RTX 3050: пауза при нагреве, передышка между пакетами
GPU_HOT_C = 75          # при этой температуре - пауза
GPU_COOL_C = 65         # продолжать после остывания до этой
PAUSE_BETWEEN = 0.3     # с, после каждого запроса к сервису эмбеддингов
MAX_UPSERT_BYTES = 8 * 1024 * 1024  # qdrant рвёт соединение на запросах > 32 МБ

DOCS = "C:/Users/user/Documents"
SOURCES = {
    "roznica_config":     f"{DOCS}/Проект Новая Розница Садыхан",
    "sadykhan_roznica":   f"{DOCS}/Садыхан",
    "erp_config":         f"{DOCS}/ЕРП_Проект/ЕРПВыгрузка",
    "erp_sadykhan":       f"{DOCS}/ЕРП_Проект/СадыханРасширение",
    "erp_migration":      f"{DOCS}/ЕРП_Проект/МиграцияРасширение",
    "utp_ssa":            f"{DOCS}/UTP_CCA/UTP_CCA",
    "liderplus_buhnya":   f"{DOCS}/liderplus Бухня ПРОД Хмл/liderplus Бухня ПРОД",
    "liderplus_sadykhan": f"{DOCS}/liderplus Бухня ПРОД Хмл/СадыханРасширениеБухня",
    "liderplus_sd":       f"{DOCS}/liderplus Бухня ПРОД Хмл/СДРасширениеБухня",
    "zup":                f"{DOCS}/ЗупХМЛ/Зуп",
    "zup_sadykhan":       f"{DOCS}/ЗупХМЛ/Расширения/ДоработкиСадыхан",
    "zup_tabel":          f"{DOCS}/ЗупХМЛ/Расширения/ФактическийТабель",
}
# Коллекции без исходной выгрузки: векторы пересчитываются из сохранённого payload
REEMBED_ONLY = ["sadykhan_config"]


def scroll_all(coll, fields):
    pts, off = [], None
    while True:
        r = requests.post(f"{QDRANT}/collections/{coll}/points/scroll", json={
            "limit": 1000, "offset": off, "with_payload": fields, "with_vector": False}).json()["result"]
        pts += r["points"]
        off = r.get("next_page_offset")
        if not off:
            return pts


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None  # нет nvidia-smi - работаем без контроля температуры


def cool_down():
    t = gpu_temp()
    if t is None or t < GPU_HOT_C:
        return
    print(f"   видеокарта {t} C - пауза до {GPU_COOL_C} C", flush=True)
    while True:
        time.sleep(5)
        t = gpu_temp()
        if t is None or t <= GPU_COOL_C:
            return


def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        cool_down()
        r = requests.post(f"{EMBED}/embed", json={"texts": texts[i:i + batch], "task": "retrieval.passage"}, timeout=300)
        r.raise_for_status()
        out += r.json()["embeddings"]
        time.sleep(PAUSE_BETWEEN)
    return out


def upsert(coll, points):
    """Пишет точки пакетами не больше MAX_UPSERT_BYTES."""
    chunk, size = [], 0
    for p in points:
        n = len(json.dumps(p, ensure_ascii=False).encode("utf-8"))
        if chunk and size + n > MAX_UPSERT_BYTES:
            requests.put(f"{QDRANT}/collections/{coll}/points?wait=true", json={"points": chunk}).raise_for_status()
            chunk, size = [], 0
        chunk.append(p)
        size += n
    if chunk:
        requests.put(f"{QDRANT}/collections/{coll}/points?wait=true", json={"points": chunk}).raise_for_status()


def recreate(coll, dim):
    requests.delete(f"{QDRANT}/collections/{coll}")
    requests.put(f"{QDRANT}/collections/{coll}", json={"vectors": {
        "object_name": {"size": dim, "distance": "Cosine", "on_disk": True},
        "friendly_name": {"size": dim, "distance": "Cosine", "on_disk": True}}}).raise_for_status()
    # индекс для фильтра по типу в MCP-поиске
    requests.put(f"{QDRANT}/collections/{coll}/index", json={
        "field_name": "object_type", "field_schema": "keyword"}).raise_for_status()


def upload(coll, rows, dim):
    """rows: [(obj_name, obj_type, synonym, doc, file_name)]"""
    recreate(coll, dim)
    for i in range(0, len(rows), 200):
        part = rows[i:i + 200]
        friendly = [f"{t}: {s}" if s else t for _, t, s, _, _ in part]
        ov = embed([f"{n} {camel_words(n)}" for n, _, _, _, _ in part])
        fv = embed(friendly)
        points = [{"id": str(uuid.uuid4()),
                   "vector": {"object_name": ov[j], "friendly_name": fv[j]},
                   "payload": {"object_name": part[j][0], "object_type": part[j][1], "doc": part[j][3],
                               "file_name": part[j][4], "friendly_name": friendly[j]}}
                  for j in range(len(part))]
        upsert(coll, points)


def main(only):
    dim = requests.get(f"{EMBED}/model-info").json()["dimensions"]
    print(f"модель: {requests.get(EMBED + '/model-info').json()['model_name']}, dim={dim}", flush=True)
    existing = {c["name"] for c in requests.get(f"{QDRANT}/collections").json()["result"]["collections"]}
    for coll, src in SOURCES.items():
        if only and coll not in only:
            continue
        t0 = time.time()
        objs = ix.collect_objects(src)
        rows = [(n, t, s, doc, f) for n, t, s, f, doc in objs]
        if coll in existing:
            old = {p["payload"].get("object_name") for p in scroll_all(coll, ["object_name"])}
            new = {r[0] for r in rows}
            overlap = len(old & new) / max(len(old), 1)
            if overlap < MIN_OVERLAP:
                print(f"ПРОПУСК {coll}: совпало {overlap:.0%} старых имён - источник не тот? {src}", flush=True)
                continue
        else:
            old, overlap = set(), 1.0
        upload(coll, rows, dim)
        cnt = requests.get(f"{QDRANT}/collections/{coll}").json()["result"]["points_count"]
        print(f"{coll:20} было={len(old):5} стало={cnt:5} с_синонимом={sum(1 for r in rows if r[2]):5} "
              f"совпадение={overlap:.0%} {time.time() - t0:.0f}s", flush=True)
    for coll in REEMBED_ONLY:
        if (only and coll not in only) or coll not in existing:
            continue
        t0 = time.time()
        pts = scroll_all(coll, True)
        rows = []
        for p in pts:
            pl = p["payload"]
            fr = pl.get("friendly_name", "")
            syn = fr.split(": ", 1)[1] if ": " in fr else ""
            rows.append((pl.get("object_name", ""), pl.get("object_type", ""), syn, pl.get("doc", ""), pl.get("file_name", "")))
        upload(coll, rows, dim)
        print(f"{coll:20} пересчитано из payload: {len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
