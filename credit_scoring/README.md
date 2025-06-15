# Credit Scoring Tool

This project provides a simple logistic regression based credit scoring model
and a FastAPI service for predicting credit scores.

## Training

Prepare a CSV file with features and a `target` column indicating default or
non-default. Run the training script:

```bash
python train.py path/to/data.csv --model-out credit_model.pkl
```

### Generating synthetic data

If you do not have real training data, you can generate a synthetic dataset for
experimentation:

```bash
python generate_synthetic_data.py --samples 1000 --out synthetic_data.csv
```

This will create a `synthetic_data.csv` file which can then be used with the
training script.

## Running the API

Install dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Use `curl` or any HTTP client to send POST requests to `/score`:

```bash
curl -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"income": 50000, "age": 30, "loan_amount": 10000, "existing_loans": 1}'
```

## Docker Deployment

A `Dockerfile` is provided for containerized deployment:

```bash
docker build -t credit_scoring .
docker run -p 8000:8000 credit_scoring
```

This will expose the API at `http://localhost:8000/score`.
