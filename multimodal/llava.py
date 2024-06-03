import time
import ollama

class Llava:
    def __init__(self):
        pass

    def get_output(self, prompt, image):
        retries = 10
        for attempt in range(retries):
            try:
                res = ollama.chat(
                    model="llava",
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image]
                    }]
                )
                return res["message"]["content"]
            except ollama.ResponseError as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    raise e

if __name__ == "__main__":
    model = Llava()
    prompt = "この画像について説明して"
    image = "./current_state.png"
    res = model.get_output(prompt, image)
    print(res)