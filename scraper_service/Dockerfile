FROM python:3.12-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium + all its OS dependencies
RUN playwright install --with-deps chromium

# Fonts for image rendering + increase shared memory for Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Chromium needs /dev/shm — Docker default is 64MB which causes crashes
ENV PYTHONUNBUFFERED=1

# Copy crawler + service code
COPY sofascore_formations_crawler.py ./sofascore_formations_crawler.py
COPY main.py ./main.py

ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
