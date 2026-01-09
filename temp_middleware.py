@app.middleware("http")
async def remove_frame_options(request, call_next):
    response = await call_next(request)
    response.headers.pop("X-Frame-Options", None)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response
