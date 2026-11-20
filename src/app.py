import uvicorn
from fastapi import FastAPI, Request
# from dialog_process import DialogProcess
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

# 初始化组件
# dialog_process = DialogProcess()

# 创建 FastAPI 实例
app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="templates"))

# 首页
@app.get("/")
async def homepage():
    return RedirectResponse("/static/index.html")

# 处理用户消息
@app.post("/chat")
async def handle_message(request: Request):
    data = await request.json()
    user_message = data["message"]
    # resp = dialog_process(user_message)  # {request: ,message: }
    return JSONResponse(user_message)

if __name__ == "__main__":
 uvicorn.run("app:app", host="127.0.0.1", port=8089, reload=True)
