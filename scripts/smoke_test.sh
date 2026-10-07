#!/usr/bin/env bash
set -e

echo "→ Health check"
curl -sf http://localhost:8000/health | grep -q '"status":"ok"' && echo "  OK"

echo "→ Model info"
curl -sf http://localhost:8000/model-info > /dev/null && echo "  OK"

echo "→ Prediction (setosa)"
curl -sf -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
  | grep -q '"class_name":"setosa"' && echo "  OK"

echo "→ MLflow reachable"
curl -sf http://localhost:5000/health > /dev/null && echo "  OK"

echo ""
echo "All checks passed."#!/usr/bin/env bash
set -e

echo "→ Health check"
curl -sf http://localhost:8000/health | grep -q '"status":"ok"' && echo "  OK"

echo "→ Model info"
curl -sf http://localhost:8000/model-info > /dev/null && echo "  OK"

echo "→ Prediction (setosa)"
curl -sf -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
  | grep -q '"class_name":"setosa"' && echo "  OK"

echo "→ MLflow reachable"
curl -sf http://localhost:5000/health > /dev/null && echo "  OK"

echo ""
echo "All checks passed."
