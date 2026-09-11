FROM  python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY download_weights.py .
RUN python download_weights.py
ENV HF_HUB_OFFLINE=1
COPY . .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
