import argparse
import os
import re
import sys
import time

import pandas as pd
from elasticsearch import Elasticsearch, helpers


def sanitize_name(path: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", path)
    return name.lower().strip("_")


def load_vectors_and_strings(vec_file: str, str_file: str):
    with open(vec_file) as vf, open(str_file) as sf:
        vectors = [[float(x) for x in line.split()] for line in vf]
        strings = [line.rstrip("\n") for line in sf]

    n = min(len(vectors), len(strings))
    return vectors[:n], strings[:n]


def load_queries(vec_query_file: str, str_query_file: str, k_query_file: str):
    with open(vec_query_file) as vf, open(str_query_file) as sf, open(k_query_file) as kf:
        vectors = [[float(x) for x in line.split()] for line in vf]
        strings = [line.rstrip("\n") for line in sf]
        ks = [int(line.strip()) for line in kf]

    n = min(len(vectors), len(strings), len(ks))
    return vectors[:n], strings[:n], ks[:n]


def load_ground_truth(path: str):
    gt = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            gt.append([int(x) for x in line.split()] if line else [])
    return gt


def escape_wildcard_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def build_client(args):
    kwargs = {"hosts": [args.host], "request_timeout": args.request_timeout}
    if args.username:
        kwargs["basic_auth"] = (args.username, args.password or "")
    return Elasticsearch(**kwargs)


def ensure_index(es: Elasticsearch, index_name: str, vectors, strings, rebuild: bool):
    if rebuild and es.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    if es.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Skipping data insertion.")
        return

    if not vectors:
        raise ValueError("No vectors loaded.")

    dim = len(vectors[0])
    print(f"Creating index {index_name} (dim={dim})...")
    es.indices.create(
        index=index_name,
        settings={
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        mappings={
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "l2_norm",
                },
                "text": {"type": "text"},
                "text_keyword": {"type": "keyword", "ignore_above": 32766},
            }
        },
    )

    print("Bulk indexing documents...")
    actions = (
        {
            "_index": index_name,
            "_id": i,  # keep 0-based ids to match ground truth convention
            "_source": {"embedding": vectors[i], "text": strings[i], "text_keyword": strings[i]},
        }
        for i in range(len(vectors))
    )
    ok, failed = 0, 0
    for success, _ in helpers.streaming_bulk(es, actions, chunk_size=500, max_retries=2):
        if success:
            ok += 1
        else:
            failed += 1
    print(f"Indexed {ok} documents, failed {failed}.")
    if failed:
        raise RuntimeError(f"Failed to index {failed} documents.")

    es.indices.put_settings(index=index_name, settings={"refresh_interval": "1s"})
    es.indices.refresh(index=index_name)
    print("Index ready.")


def run_queries(es: Elasticsearch, index_name: str, qvecs, qstrs, qks, ground_truth, candidates_list):
    print("Executing queries...")
    recalls = []
    times_us = []

    n = min(len(qvecs), len(qstrs), len(qks), len(ground_truth))
    qvecs, qstrs, qks, ground_truth = qvecs[:n], qstrs[:n], qks[:n], ground_truth[:n]

    for num_candidates in candidates_list:
        print(f"num_candidates={num_candidates}")
        total_time = 0.0
        total_recall = 0.0
        effective = 0

        for i, (qvec, qstr, qk) in enumerate(zip(qvecs, qstrs, qks)):
            if qk <= 0:
                continue

            start = time.time()
            wildcard_value = f"*{escape_wildcard_value(qstr)}*"
            resp = es.search(
                index=index_name,
                source=False,
                knn={
                    "field": "embedding",
                    "query_vector": qvec,
                    "k": qk,
                    "num_candidates": max(num_candidates, qk),
                    "filter": {
                        "wildcard": {
                            "text_keyword": {
                                "value": wildcard_value
                            }
                        }
                    },
                },
            )
            elapsed = time.time() - start
            total_time += elapsed

            hits = resp.get("hits", {}).get("hits", [])
            neighbor_ids = [int(hit["_id"]) for hit in hits]

            true_ids = set(ground_truth[i])
            if true_ids:
                effective += 1
                total_recall += len(set(neighbor_ids) & true_ids) / len(true_ids)

        avg_time = (total_time / n) * 1e6 if n else 0.0
        avg_recall = (total_recall / effective) if effective else 0.0
        times_us.append(avg_time)
        recalls.append(avg_recall)
        print(f"Average Time: {avg_time:.2f} us, Average Recall: {avg_recall:.4f}")

    return times_us, recalls


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Elasticsearch vector search with substring constraints."
    )
    parser.add_argument("string_data_file")
    parser.add_argument("vector_data_file")
    parser.add_argument("string_query_file")
    parser.add_argument("vector_query_file")
    parser.add_argument("k_file")
    parser.add_argument("ground_truth_file")
    parser.add_argument("--host", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--username", default=os.getenv("ELASTICSEARCH_USERNAME"))
    parser.add_argument("--password", default=os.getenv("ELASTICSEARCH_PASSWORD"))
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the index.")
    return parser.parse_args()


def main():
    args = parse_args()
    es = build_client(args)

    try:
        info = es.info()
        version = info.get("version", {}).get("number", "unknown")
        print(f"Connected to Elasticsearch {version}")
    except Exception as exc:
        print(f"Failed to connect to Elasticsearch at {args.host}: {exc}")
        sys.exit(1)

    print("Loading data...")
    vectors, strings = load_vectors_and_strings(args.vector_data_file, args.string_data_file)
    index_name = args.index_name or f"vectormaton_{sanitize_name(args.vector_data_file)}"
    ensure_index(es, index_name, vectors, strings, args.rebuild)
    print("Done loading data.")

    print("Loading queries and ground truth...")
    qvecs, qstrs, qks = load_queries(args.vector_query_file, args.string_query_file, args.k_file)
    ground_truth = load_ground_truth(args.ground_truth_file)

    candidates_list = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    times_us, recalls = run_queries(es, index_name, qvecs, qstrs, qks, ground_truth, candidates_list)

    print("Write statistics to file...")
    df = pd.DataFrame(
        {"num_candidates": candidates_list, "time_us": times_us, "recall": recalls}
    )
    df.to_csv("elasticsearch_hnsw_stats.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
