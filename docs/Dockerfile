# Stage 1: Build the Next.js static site with Python API docs
FROM node:18-alpine

# Install dependencies
RUN apk add --no-cache git python3 py3-pip curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv $HOME/.local/bin/uv /usr/local/bin/

# Clone the website repo
WORKDIR /website
RUN git clone --branch dev/next --single-branch --depth 1 https://gitlab.com/zrlf/bamboost-docs.git . 
RUN uv venv

# Copy the entrypoint script
COPY .entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
