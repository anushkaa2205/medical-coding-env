FROM python:3.10

WORKDIR /app

# copy everything
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ensure imports work
ENV PYTHONPATH=/app

# run server
CMD ["uvicorn", "med_env.app:app", "--host", "0.0.0.0", "--port", "8000"]