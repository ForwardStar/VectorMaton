import sys

def compute_recall(ground_truth, prediction):
    """
    Compute recall = |ground_truth ∩ prediction| / |ground_truth|
    """
    if not ground_truth:  # avoid division by zero
        return 1.0
    intersection = ground_truth.intersection(prediction)
    return len(intersection) / len(ground_truth)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_recall.py <standard_output_file> <evaluating_output_file>")
        sys.exit(1)

    std_file, eval_file = sys.argv[1], sys.argv[2]

    recalls = []

    with open(std_file, 'r') as f_std, open(eval_file, 'r') as f_eval:
        for line_std, line_eval in zip(f_std, f_eval):
            ground_truth = set(map(int, line_std.strip().split()))
            prediction = set(map(int, line_eval.strip().split()))
            recalls.append(compute_recall(ground_truth, prediction))

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"Average Recall: {avg_recall:.4f}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
