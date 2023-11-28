from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simplet5 import SimpleT5

model = SimpleT5()
# model.from_pretrained(model_type="t5", model_name="t5-base")

model.load_model("t5","D:/AstroChat/chatbot-20231123T070311Z-001/chatbot/outputs/simplet5-epoch-2-train-loss-1.7183-val-loss-1.1546", use_gpu=False)


app = FastAPI()

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)



@app.get("/echo/{text}")
async def echo_text(text: str):
  print(text)
  t = model.predict(text)
  return f"{t[0]}"