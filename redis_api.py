from typing import List, Dict, Any, Optional
import json

import redis

from config import REDIS_HOST, REDIS_PORT

client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

def get_all_images_ids() -> List[int]:
    ids = client.hkeys("images")
    # all_data = client.hgetall("data")
    # result = []
    # for i in all_images:
    #     result = {
    #         "image_bytes": all_images[i],
    #         "image_data": all_data[i],
    #         "id": i
    #     }
    result = sorted([int(i) for i in ids])
    return result

def get_image_data(image_id: int) -> str:
    data = client.hget("data", image_id).decode()
    return data

def get_image_file(image_id: int) -> bytes:
    image = client.hget("images", image_id)
    return image

def add_image(image_bytes: bytes, image_data: str, image_id: Optional[int] = 0) -> None:
    client.hset("images", image_id, image_bytes)
    client.hset("data", image_id, image_data)

def clear_all() -> None:
    client.delete("images")
    client.delete("data")