import argparse
import ctypes
import glob
import mmap
import os
import re
import sys
import threading
import time
from urllib.parse import urlparse

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


def unavailable_process_memory_stats(process_pid):
    return {
        "es_process_pid": process_pid,
        "es_process_vm_size_bytes": None,
        "es_process_rss_bytes": None,
        "es_process_rss_high_watermark_bytes": None,
        "es_process_rss_anon_bytes": None,
        "es_process_rss_file_bytes": None,
        "es_process_rss_shmem_bytes": None,
        "es_process_pss_bytes": None,
    }


def unavailable_index_file_memory_stats():
    return {
        "es_index_file_count": 0,
        "es_index_file_total_bytes": 0,
        "es_index_file_resident_bytes": 0,
        "es_index_file_resident_percent": 0.0,
    }


def collect_process_memory_stats(process_pid):
    memory_stats = unavailable_process_memory_stats(process_pid)
    if process_pid is None:
        return memory_stats

    status_path = f"/proc/{process_pid}/status"
    keys = {
        "VmSize": "es_process_vm_size_bytes",
        "VmRSS": "es_process_rss_bytes",
        "VmHWM": "es_process_rss_high_watermark_bytes",
        "RssAnon": "es_process_rss_anon_bytes",
        "RssFile": "es_process_rss_file_bytes",
        "RssShmem": "es_process_rss_shmem_bytes",
    }

    try:
        with open(status_path) as status_file:
            for line in status_file:
                name, _, value = line.partition(":")
                if name not in keys:
                    continue
                parts = value.strip().split()
                if parts:
                    memory_stats[keys[name]] = int(parts[0]) * 1024
    except OSError as exc:
        print(f"Could not read Elasticsearch process memory from {status_path}: {exc}")

    smaps_rollup_path = f"/proc/{process_pid}/smaps_rollup"
    try:
        with open(smaps_rollup_path) as smaps_file:
            for line in smaps_file:
                name, _, value = line.partition(":")
                if name == "Pss":
                    parts = value.strip().split()
                    if parts:
                        memory_stats["es_process_pss_bytes"] = int(parts[0]) * 1024
                    break
    except OSError:
        pass

    return memory_stats


def get_index_uuid(es: Elasticsearch, index_name: str):
    if not es.indices.exists(index=index_name):
        return None

    settings = es.indices.get_settings(index=index_name)
    index_settings = settings.get(index_name, {}).get("settings", {}).get("index", {})
    return index_settings.get("uuid")


def get_elasticsearch_data_paths(es: Elasticsearch, data_path_arg):
    if data_path_arg:
        return [path for path in data_path_arg.split(",") if path]

    paths = []
    try:
        stats = es.nodes.stats(metric="fs")
        for node in stats.get("nodes", {}).values():
            for data_info in node.get("fs", {}).get("data", []):
                path = data_info.get("path")
                if path:
                    paths.append(path)
    except Exception as exc:
        print(f"Could not discover Elasticsearch data paths from node stats: {exc}")

    return sorted(set(paths))


def find_index_directories(data_paths, index_uuid):
    if not index_uuid:
        return []

    index_dirs = []
    for data_path in data_paths:
        candidates = [
            os.path.join(data_path, "nodes", "*", "indices", index_uuid),
            os.path.join(data_path, "indices", index_uuid),
            os.path.join(data_path, "**", "indices", index_uuid),
        ]
        for pattern in candidates:
            for path in glob.glob(pattern, recursive=True):
                if os.path.isdir(path):
                    index_dirs.append(path)

    return sorted(set(index_dirs))


def resident_bytes_for_file(path, page_size):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0

    fd = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)

    try:
        page_count = (size + page_size - 1) // page_size
        vec = (ctypes.c_ubyte * page_count)()
        mapped_byte = ctypes.c_char.from_buffer(mapping)
        address = ctypes.addressof(mapped_byte)
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vec)
        del mapped_byte

        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno), path)

        resident_pages = sum(1 for value in vec if value & 1)
        resident_bytes = min(resident_pages * page_size, size)
        return size, resident_bytes
    finally:
        mapping.close()


def collect_index_file_memory_stats(es: Elasticsearch, index_name: str, data_path_arg):
    index_uuid = get_index_uuid(es, index_name)
    data_paths = get_elasticsearch_data_paths(es, data_path_arg)
    index_dirs = find_index_directories(data_paths, index_uuid)
    if not index_dirs:
        if index_uuid:
            print(
                "Could not find local Elasticsearch index files for "
                f"{index_name} ({index_uuid}); file residency stats unavailable."
            )
        return unavailable_index_file_memory_stats()

    page_size = os.sysconf("SC_PAGE_SIZE")
    total_bytes = 0
    resident_bytes = 0
    file_count = 0
    failed_files = 0

    for index_dir in index_dirs:
        for root, _, files in os.walk(index_dir):
            for filename in files:
                path = os.path.join(root, filename)
                if not os.path.isfile(path):
                    continue
                try:
                    file_bytes, file_resident_bytes = resident_bytes_for_file(path, page_size)
                except OSError:
                    failed_files += 1
                    continue
                total_bytes += file_bytes
                resident_bytes += file_resident_bytes
                file_count += 1

    if failed_files:
        print(f"Skipped {failed_files} Elasticsearch index files while measuring residency.")

    resident_percent = (resident_bytes / total_bytes * 100.0) if total_bytes else 0.0
    return {
        "es_index_file_count": file_count,
        "es_index_file_total_bytes": total_bytes,
        "es_index_file_resident_bytes": resident_bytes,
        "es_index_file_resident_percent": resident_percent,
    }


def collect_segment_memory_stats(es: Elasticsearch, index_name: str):
    stats = es.indices.stats(index=index_name, metric="segments")
    index_stats = stats.get("indices", {}).get(index_name, {})
    primaries_segments = index_stats.get("primaries", {}).get("segments", {})
    total_segments = index_stats.get("total", {}).get("segments", {})

    return {
        "es_primary_segments_memory_bytes": primaries_segments.get("memory_in_bytes"),
        "es_primary_index_writer_memory_bytes": primaries_segments.get("index_writer_memory_in_bytes"),
        "es_primary_version_map_memory_bytes": primaries_segments.get("version_map_memory_in_bytes"),
        "es_primary_fixed_bit_set_memory_bytes": primaries_segments.get("fixed_bit_set_memory_in_bytes"),
        "es_total_segments_memory_bytes": total_segments.get("memory_in_bytes"),
        "es_total_index_writer_memory_bytes": total_segments.get("index_writer_memory_in_bytes"),
        "es_total_version_map_memory_bytes": total_segments.get("version_map_memory_in_bytes"),
        "es_total_fixed_bit_set_memory_bytes": total_segments.get("fixed_bit_set_memory_in_bytes"),
    }


def collect_memory_stats(es: Elasticsearch, index_name: str, process_pid, data_path_arg):
    memory_stats = collect_process_memory_stats(process_pid)
    memory_stats.update(collect_index_file_memory_stats(es, index_name, data_path_arg))
    try:
        memory_stats.update(collect_segment_memory_stats(es, index_name))
    except Exception as exc:
        print(f"Could not collect Elasticsearch segment memory stats: {exc}")
        memory_stats.update(
            {
                "es_primary_segments_memory_bytes": None,
                "es_primary_index_writer_memory_bytes": None,
                "es_primary_version_map_memory_bytes": None,
                "es_primary_fixed_bit_set_memory_bytes": None,
                "es_total_segments_memory_bytes": None,
                "es_total_index_writer_memory_bytes": None,
                "es_total_version_map_memory_bytes": None,
                "es_total_fixed_bit_set_memory_bytes": None,
            }
        )
    return memory_stats


def add_memory_deltas(memory_stats, baseline_stats):
    stats_with_deltas = dict(memory_stats)
    for key, value in memory_stats.items():
        if not key.endswith("_bytes"):
            continue
        baseline_value = baseline_stats.get(key)
        delta_key = key.replace("_bytes", "_delta_from_baseline_bytes")
        stats_with_deltas[delta_key] = (
            value - baseline_value
            if value is not None and baseline_value is not None
            else None
        )
    return stats_with_deltas


class MemoryPeakTracker:
    def __init__(self, collect_fn, interval=0.05):
        self.collect_fn = collect_fn
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_stats = None

    def _sample(self):
        while not self.stop_event.is_set():
            self.record()
            self.stop_event.wait(self.interval)

    def record(self):
        stats = self.collect_fn()
        if self.peak_stats is None:
            self.peak_stats = dict(stats)
            return
        for key, value in stats.items():
            if not key.endswith("_bytes") or value is None:
                continue
            previous = self.peak_stats.get(key)
            if previous is None or value > previous:
                self.peak_stats[key] = value

    def start(self):
        self.record()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.record()
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.record()
        return self.peak_stats


def host_is_local(host):
    parsed = urlparse(host)
    hostname = parsed.hostname if parsed.hostname else host
    return hostname in {"localhost", "127.0.0.1", "::1"}


def discover_elasticsearch_pid(es: Elasticsearch, args):
    if args.process_pid is not None:
        return args.process_pid

    try:
        stats = es.nodes.stats(metric="process")
        pids = [
            node.get("process", {}).get("id")
            for node in stats.get("nodes", {}).values()
            if node.get("process", {}).get("id") is not None
        ]
        unique_pids = sorted(set(pids))
        if len(unique_pids) == 1:
            return unique_pids[0]
        if len(unique_pids) > 1:
            print(
                "Multiple Elasticsearch process IDs reported by the cluster; "
                "use --process-pid to choose one."
            )
            return None
    except Exception as exc:
        print(f"Could not discover Elasticsearch process ID from node stats: {exc}")

    if not host_is_local(args.host):
        print("Elasticsearch host is not local; process memory stats unavailable.")
        return None

    pids = []
    for pid_name in os.listdir("/proc"):
        if not pid_name.isdigit():
            continue
        try:
            with open(f"/proc/{pid_name}/cmdline", "rb") as cmdline_file:
                cmdline = cmdline_file.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        if "org.elasticsearch.bootstrap.Elasticsearch" in cmdline:
            pids.append(int(pid_name))

    if len(pids) == 1:
        return pids[0]
    if len(pids) > 1:
        print("Multiple local Elasticsearch processes found; use --process-pid to choose one.")
    else:
        print("Could not find a local Elasticsearch process; process memory stats unavailable.")
    return None


def print_memory_stats(label, memory_stats):
    print(f"Elasticsearch memory stats ({label}):")
    print(f"  process pid: {memory_stats['es_process_pid']}")
    print(f"  process RSS: {format_bytes(memory_stats['es_process_rss_bytes'])}")
    print(f"  process PSS: {format_bytes(memory_stats['es_process_pss_bytes'])}")
    print(f"  process anonymous RSS: {format_bytes(memory_stats['es_process_rss_anon_bytes'])}")
    print(f"  process file RSS: {format_bytes(memory_stats['es_process_rss_file_bytes'])}")
    print(f"  process shared RSS: {format_bytes(memory_stats['es_process_rss_shmem_bytes'])}")
    if "es_process_rss_delta_from_baseline_bytes" in memory_stats:
        print(
            "  process RSS delta from baseline: "
            f"{format_bytes(memory_stats['es_process_rss_delta_from_baseline_bytes'])}"
        )
    if "es_process_pss_delta_from_baseline_bytes" in memory_stats:
        print(
            "  process PSS delta from baseline: "
            f"{format_bytes(memory_stats['es_process_pss_delta_from_baseline_bytes'])}"
        )
    print(f"  index file count: {memory_stats['es_index_file_count']}")
    print(f"  index file total bytes: {format_bytes(memory_stats['es_index_file_total_bytes'])}")
    print(f"  index file resident bytes: {format_bytes(memory_stats['es_index_file_resident_bytes'])}")
    print(f"  index file resident percent: {memory_stats['es_index_file_resident_percent']:.2f}%")
    if "es_index_file_resident_delta_from_baseline_bytes" in memory_stats:
        print(
            "  index file resident delta from baseline: "
            f"{format_bytes(memory_stats['es_index_file_resident_delta_from_baseline_bytes'])}"
        )
    print(
        "  primary segments memory: "
        f"{format_bytes(memory_stats['es_primary_segments_memory_bytes'])}"
    )
    print(
        "  primary index writer memory: "
        f"{format_bytes(memory_stats['es_primary_index_writer_memory_bytes'])}"
    )
    print(
        "  primary version map memory: "
        f"{format_bytes(memory_stats['es_primary_version_map_memory_bytes'])}"
    )
    print(
        "  primary fixed bit set memory: "
        f"{format_bytes(memory_stats['es_primary_fixed_bit_set_memory_bytes'])}"
    )


def add_build_peak_delta_columns(df, memory_stats):
    df["build_peak_memory_bytes"] = memory_stats.get(
        "es_index_file_resident_delta_from_baseline_bytes"
    )


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
        help="Local Elasticsearch process PID to use for RSS/PSS memory measurements.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Comma-separated local Elasticsearch data paths for index file residency measurements.",
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

    process_pid = discover_elasticsearch_pid(es, args)
    memory_stats_baseline = collect_memory_stats(es, index_name, process_pid, args.data_path)
    print_memory_stats("baseline after connection setup and optional rebuild deletion", memory_stats_baseline)
    build_peak_tracker = MemoryPeakTracker(lambda: collect_process_memory_stats(process_pid))
    build_peak_tracker.start()

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
    created_index = ensure_index(es, index_name, base_vectors, base_strings, False)
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

    build_peak_memory_stats = add_memory_deltas(
        build_peak_tracker.stop(),
        memory_stats_baseline,
    )
    build_file_memory_stats = add_memory_deltas(
        collect_memory_stats(es, index_name, process_pid, args.data_path),
        memory_stats_baseline,
    )
    build_peak_memory_stats.update(
        {
            key: value
            for key, value in build_file_memory_stats.items()
            if key.startswith("es_index_file_")
        }
    )
    print_memory_stats(
        "build peak",
        {**collect_memory_stats(es, index_name, process_pid, args.data_path), **build_peak_memory_stats},
    )

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
