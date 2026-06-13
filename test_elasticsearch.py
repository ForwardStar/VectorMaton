import argparse
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


def split_base_and_insertions(vectors, strings, insertion_percentage: float):
    if insertion_percentage < 0 or insertion_percentage > 100:
        raise ValueError("--insertion-percentage must be in [0, 100].")

    insertion_count = int(len(vectors) * insertion_percentage / 100)
    base_count = len(vectors) - insertion_count
    if vectors and base_count == 0:
        raise ValueError("--insertion-percentage leaves no base vectors to build the index.")

    return (
        vectors[:base_count],
        strings[:base_count],
        vectors[base_count:],
        strings[base_count:],
        base_count,
    )


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
    kwargs = {"hosts": [args.host], "request_timeout": args.request_timeout, "verify_certs": False, "ssl_show_warn": False}
    if args.username:
        kwargs["basic_auth"] = (args.username, args.password or "")
    return Elasticsearch(**kwargs)


def bulk_index_documents(es: Elasticsearch, index_name: str, vectors, strings, id_offset: int = 0):
    actions = (
        {
            "_index": index_name,
            "_id": id_offset + i,  # keep 0-based ids to match ground truth convention
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
    if failed:
        raise RuntimeError(f"Failed to index {failed} documents.")
    return ok, failed


def index_needs_rebuild(es: Elasticsearch, index_name: str) -> bool:
    if not es.indices.exists(index=index_name):
        return True

    health = es.cluster.health(index=index_name).body
    status = health.get("status")
    if status == "red":
        print(f"Index {index_name} exists but health is red. Rebuilding it.")
        return True

    return False


def ensure_index(es: Elasticsearch, index_name: str, vectors, strings, rebuild: bool):
    if rebuild and es.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)

    if index_needs_rebuild(es, index_name):
        if es.indices.exists(index=index_name):
            print(f"Deleting unusable index: {index_name}")
            es.indices.delete(index=index_name)
    else:
        print(f"Index {index_name} already exists. Skipping data insertion.")
        return False

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
    ok, failed = bulk_index_documents(es, index_name, vectors, strings)
    print(f"Indexed {ok} documents, failed {failed}.")

    es.indices.put_settings(index=index_name, settings={"refresh_interval": "1s"})
    es.indices.refresh(index=index_name)
    print("Index ready.")
    return True


def format_bytes(value):
    if value is None:
        return "unavailable"

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def raw_vector_size_bytes(vectors):
    return sum(len(vector) for vector in vectors) * 4


def raw_string_size_bytes(strings):
    return sum(len(value.encode("utf-8")) for value in strings)


def collect_index_file_size_bytes(es: Elasticsearch, index_name: str):
    try:
        if not es.indices.exists(index=index_name):
            return None
        stats = es.indices.stats(index=index_name, metric="store").body
        return (
            stats.get("_all", {})
            .get("total", {})
            .get("store", {})
            .get("size_in_bytes")
        )
    except Exception as exc:
        print(f"Could not read Elasticsearch index file size for {index_name}: {exc}")
        return None


def collect_index_footprint_stats(es: Elasticsearch, index_name: str, vectors, strings):
    vector_bytes = raw_vector_size_bytes(vectors)
    string_bytes = raw_string_size_bytes(strings)
    index_file_bytes = collect_index_file_size_bytes(es, index_name)
    if index_file_bytes is None:
        auxiliary_bytes = None
        total_bytes = None
    else:
        auxiliary_bytes = max(index_file_bytes - vector_bytes - string_bytes, 0)
        total_bytes = index_file_bytes + vector_bytes + string_bytes

    return {
        "es_raw_vector_size_bytes": vector_bytes,
        "es_raw_string_size_bytes": string_bytes,
        "es_index_file_size_bytes": index_file_bytes,
        "es_auxiliary_size_bytes": auxiliary_bytes,
        "es_estimated_index_footprint_bytes": total_bytes,
    }


def print_memory_stats(label, memory_stats):
    print(f"Elasticsearch memory stats ({label}):")
    print(f"  raw vector size: {format_bytes(memory_stats['es_raw_vector_size_bytes'])}")
    print(f"  raw string size: {format_bytes(memory_stats['es_raw_string_size_bytes'])}")
    print(f"  index file size: {format_bytes(memory_stats['es_index_file_size_bytes'])}")
    print(f"  auxiliary size estimate: {format_bytes(memory_stats['es_auxiliary_size_bytes'])}")
    print(
        "  estimated index footprint (index files + raw vectors + raw strings): "
        f"{format_bytes(memory_stats['es_estimated_index_footprint_bytes'])}"
    )


def add_build_peak_delta_columns(df, memory_stats):
    df["build_peak_memory_bytes"] = memory_stats.get("es_estimated_index_footprint_bytes")
    df["es_estimated_index_footprint_bytes"] = memory_stats.get(
        "es_estimated_index_footprint_bytes"
    )
    df["es_index_file_size_bytes"] = memory_stats.get("es_index_file_size_bytes")
    df["es_raw_vector_size_bytes"] = memory_stats.get("es_raw_vector_size_bytes")
    df["es_raw_string_size_bytes"] = memory_stats.get("es_raw_string_size_bytes")
    df["es_auxiliary_size_bytes"] = memory_stats.get("es_auxiliary_size_bytes")


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
    parser.add_argument("--host", default="https://localhost:9200")
    parser.add_argument("--username", default="elastic")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument(
        "--process-pid",
        type=int,
        default=None,
        help="Deprecated; ignored. Elasticsearch memory is estimated from index files and raw payload size.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the index.")
    parser.add_argument(
        "--insertion-percentage",
        type=float,
        default=0.0,
        help="Percentage of tail data to insert after building the index.",
    )
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

    index_name = args.index_name or f"vectormaton_{sanitize_name(args.vector_data_file)}"
    if args.rebuild and es.indices.exists(index=index_name):
        print(f"Deleting existing index before baseline: {index_name}")
        es.indices.delete(index=index_name)

    print("Loading data...")
    vectors, strings = load_vectors_and_strings(args.vector_data_file, args.string_data_file)
    base_vectors, base_strings, insertion_vectors, insertion_strings, base_count = split_base_and_insertions(
        vectors,
        strings,
        args.insertion_percentage,
    )
    if insertion_vectors:
        print(
            f"Using {len(base_vectors)} base vectors and "
            f"{len(insertion_vectors)} insertion vectors."
        )
    index_build_start = time.time()
    created_index = ensure_index(es, index_name, base_vectors, base_strings, False)
    index_build_elapsed_us = int((time.time() - index_build_start) * 1e6)
    if created_index:
        print(f"Elasticsearch index built took {index_build_elapsed_us}us ({index_build_elapsed_us // 1000000}s)")
    else:
        print(f"Elasticsearch index loaded in {index_build_elapsed_us}us ({index_build_elapsed_us // 1000000}s)")
    if not created_index:
        print("Index already existed; index footprint is measured from the existing index.")
    if insertion_vectors:
        if created_index:
            print(f"Inserting additional {len(insertion_vectors)} documents into Elasticsearch index...")
            start = time.time()
            ok, failed = bulk_index_documents(
                es,
                index_name,
                insertion_vectors,
                insertion_strings,
                id_offset=base_count,
            )
            es.indices.refresh(index=index_name)
            print(f"Inserted {ok} documents, failed {failed}, took {(time.time() - start) * 1e6:.0f} us.")
        else:
            print("Skipping insertion because the Elasticsearch index already exists.")
    print("Done loading data.")

    build_peak_memory_stats = collect_index_footprint_stats(es, index_name, vectors, strings)
    print_memory_stats("estimated index footprint", build_peak_memory_stats)

    print("Loading queries and ground truth...")
    qvecs, qstrs, qks = load_queries(args.vector_query_file, args.string_query_file, args.k_file)
    ground_truth = load_ground_truth(args.ground_truth_file)

    candidates_list = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    times_us, recalls = run_queries(es, index_name, qvecs, qstrs, qks, ground_truth, candidates_list)

    print("Write statistics to file...")
    df = pd.DataFrame(
        {"num_candidates": candidates_list, "time_us": times_us, "recall": recalls}
    )
    add_build_peak_delta_columns(df, build_peak_memory_stats)
    df.to_csv("elasticsearch_hnsw_stats.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
