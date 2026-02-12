FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — match pipecat-ai version from main server
COPY tests/requirements.txt /app/tests/requirements.txt
RUN pip install --no-cache-dir -r /app/tests/requirements.txt

# Copy test audio files (real speech WAVs for accurate VAD/STT testing)
COPY tests/test_audio/ /app/tests/test_audio/

# Copy ALL test scripts
COPY tests/test_rtvi.py /app/tests/
COPY tests/test_text_mode.py /app/tests/
COPY tests/test_classroom_mode.py /app/tests/
COPY tests/test_performance.py /app/tests/
COPY tests/test_language_adherence.py /app/tests/
COPY tests/test_pipeline_direct.py /app/tests/
COPY tests/test_client.py /app/tests/
COPY tests/test_bargein.py /app/tests/
COPY tests/test_components.py /app/tests/
COPY tests/run_all_tests.py /app/tests/

# Copy server code (needed for unit tests that import classroom/database/translator modules)
COPY classroom.py /app/
COPY database.py /app/
COPY translator.py /app/
COPY auth.py /app/
COPY curriculum_manager.py /app/
COPY bot.py /app/
COPY server.py /app/
COPY services/ /app/services/
COPY prompts/ /app/prompts/
COPY content/ /app/content/

ENTRYPOINT ["python"]
CMD ["tests/run_all_tests.py"]
