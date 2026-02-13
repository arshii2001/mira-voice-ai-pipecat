FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY soak_test/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy soak test files
COPY soak_test/__init__.py /app/soak_test/__init__.py
COPY soak_test/personas.py /app/soak_test/personas.py
COPY soak_test/run_soak_test.py /app/soak_test/run_soak_test.py

# Create output directory
RUN mkdir -p /app/soak_test/output

ENTRYPOINT ["python", "soak_test/run_soak_test.py"]
