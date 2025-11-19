import multiprocessing

bind = "0.0.0.0:8000"
workers = max(2, multiprocessing.cpu_count() // 2)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180
keepalive = 75
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = "info"
