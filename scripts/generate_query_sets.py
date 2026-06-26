#!/usr/bin/env python3
import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASETS = [
    "spam",
    "words",
    "mtg",
    "arxiv-small",
    "swissprot",
    "code_search_net",
]
DEFAULT_PATTERN_LENGTHS = [2, 3, 4]
VECTOR_SHIFT_RANGE = 0.01


def stable_seed(*parts):
    key = ":".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


def load_strings(path):
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def generate_string_queries(strings, pattern_length, num_queries, rng):
    candidates = [string for string in strings if len(string) >= pattern_length]
    if not candidates:
        raise ValueError(f"No strings long enough for pattern length {pattern_length}.")

    queries = []
    for _ in range(num_queries):
        base_string = rng.choice(candidates)
        start = rng.randint(0, len(base_string) - pattern_length)
        queries.append(base_string[start:start + pattern_length])
    return queries


def choose_vector_indices(num_vectors, pattern_lengths, num_queries, base_seed, dataset):
    indices_by_length = {}
    needed = defaultdict(list)

    for pattern_length in pattern_lengths:
        rng = random.Random(stable_seed(base_seed, dataset, pattern_length, "vectors"))
        indices = [rng.randrange(num_vectors) for _ in range(num_queries)]
        indices_by_length[pattern_length] = indices
        for query_index, vector_index in enumerate(indices):
            needed[vector_index].append((pattern_length, query_index))

    return indices_by_length, needed


def load_selected_vectors(path, needed):
    vectors_by_length = defaultdict(dict)
    pending = dict(needed)

    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            assignments = pending.pop(line_index, None)
            if assignments is None:
                continue

            vector = list(map(float, line.split()))
            for pattern_length, query_index in assignments:
                vectors_by_length[pattern_length][query_index] = list(vector)

            if not pending:
                break

    if pending:
        first_missing = min(pending)
        raise ValueError(f"Vector file ended before selected line {first_missing}: {path}")

    return vectors_by_length


def apply_vector_shifts(vectors_by_length, pattern_lengths, num_queries, base_seed, dataset):
    for pattern_length in pattern_lengths:
        rng = random.Random(stable_seed(base_seed, dataset, pattern_length, "shift"))
        for query_index in range(num_queries // 2, num_queries):
            vector = vectors_by_length[pattern_length][query_index]
            vectors_by_length[pattern_length][query_index] = [
                value + rng.uniform(-VECTOR_SHIFT_RANGE, VECTOR_SHIFT_RANGE)
                for value in vector
            ]


def save_queries(output_dir, string_queries, vector_queries, k):
    output_dir.mkdir(parents=True, exist_ok=True)
    strings_path = output_dir / "strings.txt"
    vectors_path = output_dir / "vectors.txt"
    k_path = output_dir / "k.txt"

    with strings_path.open("w", encoding="utf-8") as sf, \
            vectors_path.open("w", encoding="utf-8") as vf, \
            k_path.open("w", encoding="utf-8") as kf:
        for string_query, vector_query in zip(string_queries, vector_queries):
            sf.write(string_query + "\n")
            vf.write(" ".join(map(str, vector_query)) + "\n")
            kf.write(f"{k}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save query sets for multiple datasets and pattern lengths."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names under datasets/. Defaults to the recall/QPS datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Use every dataset directory that has strings.txt and vectors.txt.",
    )
    parser.add_argument(
        "--pattern-lengths",
        "-p",
        nargs="+",
        type=int,
        default=DEFAULT_PATTERN_LENGTHS,
        help="Pattern lengths to generate. Default: 2 3 4.",
    )
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-dir", default="datasets")
    parser.add_argument("--output-dir", default="queries")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_dir)
    output_root = Path(args.output_dir)

    if args.all_datasets:
        datasets = sorted(
            path.name
            for path in dataset_root.iterdir()
            if (path / "strings.txt").exists() and (path / "vectors.txt").exists()
        )
    else:
        datasets = args.datasets

    for dataset in datasets:
        string_path = dataset_root / dataset / "strings.txt"
        vector_path = dataset_root / dataset / "vectors.txt"
        if not string_path.exists() or not vector_path.exists():
            raise FileNotFoundError(f"Missing strings.txt or vectors.txt for dataset: {dataset}")

        strings = load_strings(string_path)
        indices_by_length, needed = choose_vector_indices(
            len(strings), args.pattern_lengths, args.num_queries, args.seed, dataset
        )
        vectors_by_length = load_selected_vectors(vector_path, needed)
        apply_vector_shifts(vectors_by_length, args.pattern_lengths, args.num_queries, args.seed, dataset)

        for pattern_length in args.pattern_lengths:
            string_rng = random.Random(stable_seed(args.seed, dataset, pattern_length, "strings"))
            string_queries = generate_string_queries(
                strings, pattern_length, args.num_queries, string_rng
            )
            vector_queries = [
                vectors_by_length[pattern_length][query_index]
                for query_index in range(args.num_queries)
            ]

            query_dir = output_root / dataset / str(pattern_length)
            save_queries(query_dir, string_queries, vector_queries, args.k)
            print(f"Wrote {args.num_queries} queries: {query_dir}", flush=True)


if __name__ == "__main__":
    main()
