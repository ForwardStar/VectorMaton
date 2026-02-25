import argparse
import random
import string


def random_string(rng: random.Random, min_len: int, max_len: int) -> str:
    length = rng.randint(min_len, max_len)
    alphabet = string.ascii_lowercase
    return "".join(rng.choice(alphabet) for _ in range(length))


def random_vector(rng: random.Random, dim: int) -> list[float]:
    return [rng.random() for _ in range(dim)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random dataset for scalability tests."
    )
    parser.add_argument("--num-items", type=int, default=100, help="Number of vectors/strings.")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensionality.")
    parser.add_argument("--min-str-len", type=int, default=100, help="Minimum string length.")
    parser.add_argument("--max-str-len", type=int, default=500, help="Maximum string length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--vectors-out",
        default="vectors_scalability.txt",
        help="Output path for vectors.",
    )
    parser.add_argument(
        "--strings-out",
        default="strings_scalability.txt",
        help="Output path for strings.",
    )
    args = parser.parse_args()

    if args.num_items <= 0:
        raise ValueError("--num-items must be > 0")
    if args.dim <= 0:
        raise ValueError("--dim must be > 0")
    if args.min_str_len <= 0:
        raise ValueError("--min-str-len must be > 0")
    if args.max_str_len < args.min_str_len:
        raise ValueError("--max-str-len must be >= --min-str-len")

    rng = random.Random(args.seed)

    with open(args.vectors_out, "w") as vf, open(args.strings_out, "w") as sf:
        for _ in range(args.num_items):
            vec = random_vector(rng, args.dim)
            text = random_string(rng, args.min_str_len, args.max_str_len)
            vf.write(" ".join(map(str, vec)) + "\n")
            sf.write(text + "\n")

    print(f"Wrote {args.num_items} vectors to {args.vectors_out}")
    print(f"Wrote {args.num_items} strings to {args.strings_out}")
    print(
        f"Vector dim={args.dim}, string length range=[{args.min_str_len}, {args.max_str_len}], seed={args.seed}"
    )


if __name__ == "__main__":
    main()
