from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import cv2
import asyncio
import json

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.get("/")
async def root():
    return {"message": "CCTV Analysis Server API is running"}