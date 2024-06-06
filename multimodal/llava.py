import time
import ollama

class Llava:
    def __init__(self):
        pass

    def get_output(self, prompt, image):
        res = ollama.chat(
            #model="llava",
            model="llava",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image]
            }],
            options={"num_ctx": 2048}
        )
        return res["message"]["content"]

if __name__ == "__main__":
    model = Llava()
    prompt = "この画像について説明して"
    image = "./current_state.png"
    res = model.get_output(prompt, image)
    print(res)