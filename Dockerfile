FROM python:3.12-slim

# System libraries needed by PyQt5 on Debian-based images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdbus-1-3 \
    libgl1 \
    libglib2.0-0 \
    libice6 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    libxrender1 \
    libxinerama1 \
    libxcursor1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

ENV QT_X11_NO_MITSHM=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Arithmetic_Difficulty ./Arithmetic_Difficulty
WORKDIR /app/Arithmetic_Difficulty

CMD ["python", "mainExperiment.py"]
