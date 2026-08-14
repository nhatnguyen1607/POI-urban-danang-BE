# Sử dụng Node.js làm base
FROM node:20

# Cài đặt Python và Pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file quản lý gói
COPY package*.json ./
# Trong server.js của bạn gọi 'python', nên ta tạo link cho chắc chắn
RUN ln -s /usr/bin/python3 /usr/bin/python

# Cài đặt Node dependencies
RUN npm ci --omit=dev

# Tạo file requirements.txt nếu bạn chưa có, 
# hoặc copy nó vào nếu đã có để cài đặt thư viện Python (torch, pandas...)
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
    pip3 install --no-cache-dir --default-timeout=1000 -r requirements.txt --break-system-packages; \
    fi

# Copy toàn bộ code
COPY . .

# Fail the image build when runtime data is missing, partial, or not approved.
RUN npm run data:verify

# Hugging Face Spaces chạy trên port 7860
EXPOSE 7860
ENV PORT=7860
# Tăng timeout cho Python processes
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:' + (process.env.PORT || 7860) + '/api/v2/cities').then((response) => { if (!response.ok) process.exit(1); }).catch(() => process.exit(1))"

# Lệnh khởi chạy
CMD ["npm", "start"]
