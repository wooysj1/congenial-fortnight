import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import train_model, save_model


def main(data_path, model_out='credit_model.pkl'):
    data = pd.read_csv(data_path)

    # Assume target column is named 'target'
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.2f}")

    save_model(model, model_out)
    print(f"Model saved to {model_out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train credit scoring model')
    parser.add_argument('data', help='Path to CSV training data')
    parser.add_argument('--model-out', default='credit_model.pkl')
    args = parser.parse_args()
    main(args.data, args.model_out)
