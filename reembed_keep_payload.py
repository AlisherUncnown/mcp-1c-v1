"""
Пересчёт векторов коллекции текущей моделью embedding-service БЕЗ потери данных точек.

  python reembed_keep_payload.py zup_hml utp_sp_hml its_books

Для коллекций, у которых нет исходной выгрузки или payload богаче схемы reindex_all
(поля synonym/comment, книги ИТС, стандарты): id и payload каждой точки сохраняются как есть,
меняются только векторы. Тексты векторов те же, что даёт reindex_all.upload:
  object_name   <- "<object_name> <camel_words(object_name)>"
  friendly_name <- payload.friendly_name
Сначала считаются ВСЕ векторы, потом коллекция пересоздаётся: сбой посередине не портит данные.
"""
import sys
import time
import requests

import reindex_all as ra
from upload_zip import camel_words


def reembed(coll, dim):
    t0 = time.time()
    pts = ra.scroll_all(coll, True)
    names = [(p["payload"].get("object_name") or "") for p in pts]
    friendly = [(p["payload"].get("friendly_name") or "") for p in pts]
    ov = ra.embed([f"{n} {camel_words(n)}" for n in names])
    fv = ra.embed(friendly)
    if len(ov) != len(pts) or len(fv) != len(pts):
        raise RuntimeError(f"{coll}: векторов {len(ov)}/{len(fv)} при {len(pts)} точках - коллекция не тронута")
    ra.recreate(coll, dim)
    ra.upsert(coll, [{"id": p["id"], "vector": {"object_name": ov[i], "friendly_name": fv[i]}, "payload": p["payload"]}
                     for i, p in enumerate(pts)])
    after = requests.get(f"{ra.QDRANT}/collections/{coll}").json()["result"]["points_count"]
    print(f"{coll:20} было={len(pts):5} стало={after:5} {time.time() - t0:.0f}s", flush=True)


def main(colls):
    info = requests.get(f"{ra.EMBED}/model-info").json()
    print(f"модель: {info['model_name']}, dim={info['dimensions']}", flush=True)
    existing = {c["name"] for c in requests.get(f"{ra.QDRANT}/collections").json()["result"]["collections"]}
    for coll in colls:
        if coll not in existing:
            print(f"ПРОПУСК {coll}: нет такой коллекции", flush=True)
            continue
        reembed(coll, info["dimensions"])
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
