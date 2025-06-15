import pandas as pd
from sklearn.datasets import make_classification


def generate(n_samples=1000, random_state=42, out_path="synthetic_data.csv"):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    df = pd.DataFrame(
        X, columns=["income", "age", "loan_amount", "existing_loans"]
    )

    df["income"] = (df["income"] - df["income"].min()) / (
        df["income"].max() - df["income"].min()
    )
    df["income"] = df["income"] * 90000 + 10000

    df["age"] = (df["age"] - df["age"].min()) / (
        df["age"].max() - df["age"].min()
    )
    df["age"] = (df["age"] * 42 + 18).round()

    df["loan_amount"] = (df["loan_amount"] - df["loan_amount"].min()) / (
        df["loan_amount"].max() - df["loan_amount"].min()
    )
    df["loan_amount"] = df["loan_amount"] * 20000 + 1000

    df["existing_loans"] = (
        (df["existing_loans"] - df["existing_loans"].min())
        / (df["existing_loans"].max() - df["existing_loans"].min())
        * 3
    ).round()

    df["target"] = y
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic credit data"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out", default="synthetic_data.csv")
    args = parser.parse_args()
    generate(args.samples, out_path=args.out)
