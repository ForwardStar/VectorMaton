import argparse
import psycopg2
import numpy as np
import sys
import threading
import time
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark pgvector HNSW search with substring constraints."
    )
    parser.add_argument("string_data_file")
    parser.add_argument("vector_data_file")
    parser.add_argument("string_query_file")
    parser.add_argument("vector_query_file")
    parser.add_argument("k_file")
    parser.add_argument("ground_truth_file")
    parser.add_argument(
        "--insertion-percentage",
        type=float,
        default=0.0,
        help="Percentage of tail data to insert after building the index.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the table and index.")
    parser.add_argument("--write-output", default=None, help="Write returned ids for each query and ef_search value.")
    return parser.parse_args()


def split_base_and_insertions(vectors, strings, insertion_percentage):
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
    )


def format_bytes(value):
    if value is None:
        return "unavailable"

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def unavailable_memory_stats(backend_pid):
    return {
        "pg_backend_pid": backend_pid,
        "pg_backend_vm_size_bytes": None,
        "pg_backend_rss_bytes": None,
        "pg_backend_rss_high_watermark_bytes": None,
        "pg_backend_rss_anon_bytes": None,
        "pg_backend_rss_file_bytes": None,
        "pg_backend_rss_shmem_bytes": None,
    }


def collect_memory_stats(backend_pid):
    status_path = f"/proc/{backend_pid}/status"
    keys = {
        "VmSize": "pg_backend_vm_size_bytes",
        "VmRSS": "pg_backend_rss_bytes",
        "VmHWM": "pg_backend_rss_high_watermark_bytes",
        "RssAnon": "pg_backend_rss_anon_bytes",
        "RssFile": "pg_backend_rss_file_bytes",
        "RssShmem": "pg_backend_rss_shmem_bytes",
    }
    memory_stats = unavailable_memory_stats(backend_pid)

    try:
        with open(status_path) as status_file:
            for line in status_file:
                name, _, value = line.partition(":")
                if name not in keys:
                    continue
                parts = value.strip().split()
                if not parts:
                    continue
                memory_stats[keys[name]] = int(parts[0]) * 1024
    except OSError as exc:
        print(f"Could not read PostgreSQL backend memory from {status_path}: {exc}")

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


def print_memory_stats(label, memory_stats):
    print(f"pgvector PostgreSQL backend memory ({label}):")
    print(f"  backend pid: {memory_stats['pg_backend_pid']}")
    print(f"  RSS: {format_bytes(memory_stats['pg_backend_rss_bytes'])}")
    print(f"  RSS high watermark: {format_bytes(memory_stats['pg_backend_rss_high_watermark_bytes'])}")
    print(f"  anonymous RSS: {format_bytes(memory_stats['pg_backend_rss_anon_bytes'])}")
    print(f"  file RSS: {format_bytes(memory_stats['pg_backend_rss_file_bytes'])}")
    print(f"  shared RSS: {format_bytes(memory_stats['pg_backend_rss_shmem_bytes'])}")
    if "pg_backend_rss_delta_from_baseline_bytes" in memory_stats:
        print(
            "  RSS delta from baseline: "
            f"{format_bytes(memory_stats['pg_backend_rss_delta_from_baseline_bytes'])}"
        )


def add_build_peak_delta_columns(df, memory_stats):
    df["build_peak_memory_bytes"] = memory_stats.get(
        "pg_backend_rss_delta_from_baseline_bytes"
    )


args = parse_args()

conn = psycopg2.connect(
    dbname="postgres"
)
cur = conn.cursor()

str_file = args.string_data_file
vec_file = args.vector_data_file
vectors = []
strings = []
insertion_vectors = []
insertion_strings = []
n = 0
table_name = vec_file.replace('.', '_').replace('/', '_').replace('-', '_')

print("Loading data...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("SET temp_buffers = '128GB';")
conn.commit()
cur.execute("SHOW temp_buffers;")
conn.commit()
buff_size = cur.fetchall()
print(f"temp_buffers size set to: {buff_size[0][0]}")
cur.execute("SELECT pg_backend_pid();")
backend_pid = cur.fetchone()[0]

batch = [(vectors[i], strings[i].strip()) for i in range(n)]
table_exists = True
# Create table, insert data and build index if it does not exist
if args.rebuild:
    print(f"Deleting existing index and table for rebuild: {table_name}")
    cur.execute("DROP INDEX IF EXISTS idx_vec_hnsw;")
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()

memory_stats_baseline = None
build_peak_memory_stats = None

cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);", (table_name,))
if not cur.fetchone()[0]:
    with open(vec_file) as vf, open(str_file) as sf:
        vectors = vf.readlines()
        strings = sf.readlines()
        for i in range(len(vectors)):
            vectors[i] = [float(x) for x in vectors[i].split()]
        n = min(len(vectors), len(strings))
        if len(vectors) > n:
            vectors = vectors[:n]
        if len(strings) > n:
            strings = strings[:n]
    vectors, strings, insertion_vectors, insertion_strings = split_base_and_insertions(
        vectors,
        strings,
        args.insertion_percentage,
    )
    if insertion_vectors:
        print(
            f"Using {len(vectors)} base vectors and "
            f"{len(insertion_vectors)} insertion vectors."
        )
    n = len(vectors)
    table_exists = False
    dim = len(vectors[0])

    sql = f"""
    CREATE TEMP TABLE {table_name} (
        id SERIAL PRIMARY KEY,
        embedding VECTOR({dim}),
        text TEXT
    );"""
    cur.execute(sql)
    conn.commit()
    memory_stats_baseline = collect_memory_stats(backend_pid)
    print_memory_stats("baseline before data insertion", memory_stats_baseline)
    build_peak_tracker = MemoryPeakTracker(lambda: collect_memory_stats(backend_pid))
    build_peak_tracker.start()
    sql = f"""
    INSERT INTO {table_name} (embedding, text) VALUES (%s, %s);"""
    cur.executemany(sql, [(vectors[i], strings[i].strip()) for i in range(n)])
    conn.commit()
else:
    print(f"Table {table_name} already exists. Skipping data insertion.")
print("Done loading data.")

print("Building HNSW index...")
index_build_start = time.time()
if not table_exists:
    sql = f"""DROP INDEX IF EXISTS idx_vec_hnsw;"""
    cur.execute(sql)
    conn.commit()
    sql = f"""CREATE INDEX idx_vec_hnsw ON {table_name} USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);"""
    cur.execute(sql)
    conn.commit()
else:
    print(f"Index on table {table_name} may already exist. Skipping index creation.")
index_build_elapsed_us = int((time.time() - index_build_start) * 1e6)
if not table_exists:
    print(f"pgvector index built took {index_build_elapsed_us}us ({index_build_elapsed_us // 1000000}s)")
else:
    print(f"pgvector index loaded in {index_build_elapsed_us}us ({index_build_elapsed_us // 1000000}s)")
print("Done building index.")

average_insertion_time_us = None
if insertion_vectors:
    if not table_exists:
        print(f"Inserting additional {len(insertion_vectors)} rows into pgvector table...")
        start = time.time()
        sql = f"""
        INSERT INTO {table_name} (embedding, text) VALUES (%s, %s);"""
        cur.executemany(
            sql,
            [(insertion_vectors[i], insertion_strings[i].strip()) for i in range(len(insertion_vectors))]
        )
        conn.commit()
        insertion_elapsed_us = (time.time() - start) * 1e6
        average_insertion_time_us = insertion_elapsed_us / len(insertion_vectors)
        print(
            f"Insertion took {insertion_elapsed_us:.0f} us "
            f"({average_insertion_time_us:.2f} us/row)."
        )
    else:
        print("Skipping insertion because the pgvector table already exists.")

if not table_exists:
    build_peak_memory_stats = add_memory_deltas(
        build_peak_tracker.stop(),
        memory_stats_baseline,
    )
else:
    memory_stats_baseline = collect_memory_stats(backend_pid)
    build_peak_memory_stats = add_memory_deltas(memory_stats_baseline, memory_stats_baseline)
print_memory_stats("build peak", build_peak_memory_stats)

# Execute queries
str_query_file = args.string_query_file
vec_query_file = args.vector_query_file
k_query_file = args.k_file
vectors = []
strings = []
k = []
n = 0

with open(vec_query_file) as vf, open(str_query_file) as sf, open(k_query_file) as kf:
    vectors = vf.readlines()
    strings = sf.readlines()
    k = kf.readlines()
    for i in range(len(vectors)):
        vectors[i] = [float(x) for x in vectors[i].split()]
    n = min(len(vectors), len(strings), len(k))
    if len(vectors) > n:
        vectors = vectors[:n]
    if len(strings) > n:
        strings = strings[:n]
    if len(k) > n:
        k = k[:n]

ground_truth_file = args.ground_truth_file
ground_truth = []
with open(ground_truth_file) as gtf:
    for line in gtf:
        ids = [int(x) for x in line.strip().split()]
        ground_truth.append(ids)

def format_query_row(ef_search, k, neighbor_ids):
    ids = " ".join(str(value) for value in neighbor_ids)
    suffix = f": {ids}" if ids else ":"
    return f"ef_search = {ef_search}, k = {k}{suffix}"


def write_query_output(path, rows_by_query):
    if not path:
        return
    with open(path, "w") as output:
        for query_idx, rows in enumerate(rows_by_query, start=1):
            output.write(f"Query {query_idx}:\n")
            for row in rows:
                output.write(row + "\n")


print("Executing queries...")
recall = []
time_taken = []
query_output_rows = [[] for _ in range(n)]
ef_search_values = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1000]
for ef_search in ef_search_values:
    print(f"ef_search={ef_search}")
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    total_time = 0.0
    total_recall = 0.0
    effective = 0
    for i, (qvec, qstr, qk) in enumerate(zip(vectors, strings, k)):
        start = time.time()

        # PostgreSQL query: substring filter + vector distance
        sql = f"""
            SELECT id
            FROM {table_name}
            WHERE text LIKE %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        cur.execute(
            sql,
            (f"%{qstr.strip()}%", qvec, int(qk.strip()))
        )

        rows = cur.fetchall()
        elapsed = time.time() - start
        total_time += elapsed

        neighbor_ids = [row[0] - 1 for row in rows]
        query_output_rows[i].append(format_query_row(ef_search, int(qk.strip()), neighbor_ids))

        true_ids = set(ground_truth[i])
        if len(true_ids) > 0:
            effective += 1
            total_recall += len(set(neighbor_ids) & true_ids) / len(true_ids)

    avg_time = total_time / n * 1e6  # convert to microseconds
    avg_recall = total_recall / effective
    time_taken.append(avg_time)
    recall.append(avg_recall)
    print(f"Average Time: {avg_time} us, Average Recall: {avg_recall:.4f}")

write_query_output(args.write_output, query_output_rows)
print("Query done.")
print("Write statistics to file...")
df = pd.DataFrame({
    'ef_search': ef_search_values,
    'time_us': time_taken,
    'recall': recall
})
df["index_build_time_us"] = index_build_elapsed_us
df["average_insertion_time_us"] = average_insertion_time_us
add_build_peak_delta_columns(df, build_peak_memory_stats)
df.to_csv("pgvector_hnsw_stats.csv", index=False)
print("Done.")
