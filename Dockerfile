# --- Builder Stage ---
FROM python:3.13.2-slim-bookworm AS builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install necessary packages and remove apt caches in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /code

# Copy only dependency files and install dependencies.
COPY pyproject.toml poetry.lock /code/
RUN poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-interaction --no-ansi && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /root/.cache

# Copy SSH configuration and all code
COPY sshd_config /etc/ssh/
COPY . /code

# --- Final (Runtime) Stage ---
FROM python:3.13.2-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the entire /usr/local from builder to capture all installed executables and libraries
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /etc/ssh/sshd_config /etc/ssh/sshd_config
COPY --from=builder /code /code

# Set root password for SSH access
RUN echo "root:Docker!" | chpasswd

WORKDIR /code

EXPOSE 8000 2222

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
