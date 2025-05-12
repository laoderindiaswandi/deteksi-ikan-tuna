# Gunakan base image yang ringan tapi lengkap
FROM python:3.10-slim

# Install dependensi sistem
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori kerja
WORKDIR /app

# Salin requirements.txt dulu sebelum install
COPY requirements.txt .

# Install pip, wheel, setuptools terbaru dan dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install PyTorch CPU + torchvision
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install detectron2 yang cocok untuk CPU
RUN pip install git+https://github.com/facebookresearch/detectron2.git@main

# Salin semua file ke dalam container
COPY . .

# Jalankan aplikasi Flask
CMD ["python", "-u", "app.py"]

