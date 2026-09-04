FROM  python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('THENDO1977/from-scratch-models', 'translation/transformer_weights.pt'); \
    hf_hub_download('THENDO1977/from-scratch-models', 'translation/medical_transformer_weights.pt'); \
    hf_hub_download('THENDO1977/from-scratch-models', 'translation/sp.model'); \
    hf_hub_download('THENDO1977/from-scratch-models', 'coding/coding_model_weights.pt'); \
    hf_hub_download('THENDO1977/from-scratch-models', 'coding/sp.model')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

