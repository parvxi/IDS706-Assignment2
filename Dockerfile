# Start from a small official image that already has Python 3.12
FROM python:3.12-slim

# Work inside /app in the container
WORKDIR /app

# Install packages first, so Docker can reuse this step
# when only my code changes (faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code, data, and tests
COPY . .

# By default, run the full analysis
CMD ["python", "EDA_Dataset.py"]