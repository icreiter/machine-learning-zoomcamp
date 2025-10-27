FROM agrigorev/zoomcamp-model:2025

# Set the working directory
WORKDIR /app

# --- Dependency Installation using uv ---
# 1. Install uv itself
RUN pip install uv

# 2. Copy the dependency configuration file
#    Only copy the file needed to define dependencies
COPY "pyproject.toml" .

# 3. Use uv sync to install dependencies from pyproject.toml
#    This installs packages compatible with the base image's Python (3.13)
# RUN uv sync
RUN uv pip install -e . --system

# --- Application Files ---
# Copy the server script and the model file in one step
COPY "app_docker.py" "pipeline_v1.bin" .

# Expose the port where the application will be listening
EXPOSE 9695

# Command to run the FastAPI application using uvicorn (Exec form)
CMD ["python", "-m", "uvicorn", "app_docker:app", "--host", "0.0.0.0", "--port", "9695"]