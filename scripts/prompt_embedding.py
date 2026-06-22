#!/usr/bin/env python3
import argparse

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
QUERY_GROUPS = [
    (
        "academia",
        [
            "How does academia support scientific research?",
            "How did modern academia develop?",
            "What careers and institutions are part of academia?",
        ],
    ),
    (
        "cosmology",
        [
            "How does cosmology explain the origin of the universe?",
            "What evidence supports modern cosmology?",
            "Which discoveries transformed physical cosmology?",
        ],
    ),
    (
        "immunology",
        [
            "How does the human immune system fight disease?",
            "What are the major fields of immunology?",
            "How do antibodies recognize infectious organisms?",
        ],
    ),
    (
        "paleontology",
        [
            "How do paleontologists reconstruct prehistoric life?",
            "What can fossils reveal about evolution?",
            "Which discoveries transformed modern paleontology?",
        ],
    ),
    (
        "vaccination",
        [
            "How does vaccination create immunity?",
            "Which vaccination campaigns eliminated major diseases?",
            "How were modern vaccines developed?",
        ],
    ),
    (
        "volcano",
        [
            "What causes volcanic eruptions?",
            "Which volcanoes produced the largest eruptions?",
            "How do scientists monitor active volcanoes?",
        ],
    ),
    (
        "microprocessor",
        [
            "How does a microprocessor execute instructions?",
            "How did microprocessor architecture evolve?",
            "Which inventions enabled modern microprocessors?",
        ],
    ),
    (
        "biodiversity",
        [
            "Why is biodiversity important to ecosystems?",
            "What are the main threats to global biodiversity?",
            "How do conservation programs protect biodiversity?",
        ],
    ),
    (
        "relativity",
        [
            "How does relativity describe gravity?",
            "What experimental evidence supports general relativity?",
            "How are space and time connected in special relativity?",
        ],
    ),
    (
        "shakespeare",
        [
            "What plays did William Shakespeare write?",
            "How did Shakespeare influence English literature?",
            "What themes recur in Shakespeare tragedies?",
        ],
    ),
    (
        "genetics",
        [
            "How are traits inherited through genetics?",
            "Which discoveries established modern genetics?",
            "How do mutations affect genes and inherited traits?",
        ],
    ),
    (
        "archaeology",
        [
            "How do archaeologists date ancient settlements?",
            "Which archaeological discoveries changed our understanding of history?",
            "What scientific methods are used in archaeology?",
        ],
    ),
    (
        "quantum",
        [
            "What are the basic principles of quantum mechanics?",
            "How does quantum theory describe particles?",
            "Which experiments established quantum physics?",
        ],
    ),
    (
        "astronomy",
        [
            "How did modern astronomy develop?",
            "Which discoveries changed our understanding of the universe?",
            "What instruments are used for astronomical observations?",
        ],
    ),
    (
        "earthquake",
        [
            "What causes earthquakes?",
            "How are earthquake magnitudes measured?",
            "Which earthquakes caused major historical disasters?",
        ],
    ),
    (
        "climate",
        [
            "What causes global climate change?",
            "How does climate change affect ecosystems?",
            "Which policies can reduce climate change?",
        ],
    ),
    (
        "automation",
        [
            "How does automation replace repetitive human work?",
            "What is the history of industrial automation?",
            "Which technologies are used for automation?",
        ],
    ),
    (
        "neural",
        [
            "How do neural networks learn from data?",
            "What are the major types of neural networks?",
            "Which applications use neural network models?",
        ],
    ),
    (
        "caesar",
        [
            "How did Julius Caesar rise to power?",
            "What military campaigns were led by Julius Caesar?",
            "How did Caesar transform Roman government?",
        ],
    ),
    (
        "solar",
        [
            "How was the Solar System formed?",
            "Which objects make up the Solar System?",
            "How are planets in the Solar System explored?",
        ],
    ),
    (
        "classical",
        [
            "How did classical music develop?",
            "Who were the most influential classical music composers?",
            "What distinguishes major classical music periods?",
        ],
    ),
    (
        "athletics",
        [
            "How did organized athletics develop?",
            "Which events are included in track and field athletics?",
            "Who are notable competitors in international athletics?",
        ],
    ),
    (
        "diplomacy",
        [
            "How does diplomacy resolve international conflicts?",
            "What is the history of modern diplomacy?",
            "Which institutions conduct international diplomacy?",
        ],
    ),
    (
        "humanitarian",
        [
            "How did international humanitarian law develop?",
            "Which organizations provide humanitarian assistance?",
            "How are humanitarian crises addressed?",
        ],
    ),
    (
        "holocaust",
        [
            "What caused the Holocaust?",
            "How was the Holocaust carried out during World War II?",
            "How is the Holocaust remembered and documented?",
        ],
    ),
]

DEFAULT_QUERIES = [
    (filter_string, prompt)
    for filter_string, prompts in QUERY_GROUPS
    for prompt in prompts
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sentence-transformer embeddings for prompt queries."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Generate one custom prompt instead of the default query batch.",
    )
    parser.add_argument(
        "--filter-string",
        help="Normalized substring filter for a custom prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Defaults to {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt and not args.filter_string:
        raise SystemExit("--filter-string is required when a custom prompt is given")
    if args.filter_string and not args.prompt:
        raise SystemExit("a custom prompt is required with --filter-string")

    queries = [(args.filter_string, args.prompt)] if args.prompt else DEFAULT_QUERIES
    model = SentenceTransformer(args.model)
    vectors = model.encode([prompt for _, prompt in queries])

    with open("vectors_prompt.txt", "w") as output_file:
        for vector in vectors:
            output_file.write(" ".join(map(str, vector.tolist())) + "\n")
    with open("string_prompt.txt", "w") as string_file:
        for filter_string, _ in queries:
            string_file.write(filter_string + "\n")
    with open("k_prompt.txt", "w") as k_file:
        for _ in range(len(queries)):
            k_file.write("10\n")

    print(f"Wrote {len(queries)} query strings to string_prompt.txt")
    print(
        f"Wrote {len(vectors)} {vectors.shape[1]}-dimensional embeddings "
        "to vectors_prompt.txt"
    )


if __name__ == "__main__":
    main()
