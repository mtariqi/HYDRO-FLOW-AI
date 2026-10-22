FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY configs ./configs
COPY docs ./docs
COPY knowledge ./knowledge
EXPOSE 8000
CMD ["hydroflow", "serve", "--artifacts", "/app/artifacts"]

