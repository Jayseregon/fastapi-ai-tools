FROM python:3.13.2-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y curl build-essential openssh-server && rm -rf /var/lib/apt/lists/*

# Install Poetry using its official installer
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set root password for SSH access
RUN echo "root:Docker!" | chpasswd

RUN mkdir -p /code
WORKDIR /code

# Copy project configuration and install dependencies with Poetry
COPY pyproject.toml poetry.lock /code/
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy sshd_config for SSH access
COPY sshd_config /etc/ssh/


# Remove build tools and caches to shrink the image while preserving installed packages
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /root/.cache

# Copy code
COPY . /code

# Expose ports for the app and SSH server
EXPOSE 8000 2222

# Production command for FastAPI
CMD ["fastapi", "run", "src/main.py", "--port", "8000"]
