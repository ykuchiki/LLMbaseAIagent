import os
import random
import numpy as np
import matplotlib.pyplot as plt
import replicate
from groq import Groq

os.environ["REPLICATE_API_TOKEN"] = "apikey for replicate"
client = Groq(
    # This is the default and can be omitted
    api_key="apikey for Grog",
)
class people_flow1:
    def __init__(self, people_num, wall_x, wall_y, dt):
        self.people_num = people_num
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.dt = dt
        self.positions = np.random.rand(people_num, 2) * [wall_x, wall_y]
        self.target = np.array([wall_x / 2, wall_y / 2])

    def __start_paint(self):
        directions = ['up', 'down', 'left', 'right']
        for i in range(self.people_num):
            direction = random.choice(directions)
            if direction == 'up' and self.positions[i][1] < self.wall_y:
                self.positions[i][1] = min(self.positions[i][1] + self.dt, self.wall_y)
            elif direction == 'down' and self.positions[i][1] > 0:
                self.positions[i][1] = max(self.positions[i][1] - self.dt, 0)
            elif direction == 'left' and self.positions[i][0] > 0:
                self.positions[i][0] = max(self.positions[i][0] - self.dt, 0)
            elif direction == 'right' and self.positions[i][0] < self.wall_x:
                self.positions[i][0] = min(self.positions[i][0] + self.dt, self.wall_x)

    def update_position_based_on_prompt(self):
        for i in range(self.people_num):
            # Create a prompt based on current and target positions
            prompt = f"Agent at ({self.positions[i][0]:.2f}, {self.positions[i][1]:.2f}), target at ({self.target[0]:.2f}, {self.target[1]:.2f}). Choose: up, down, right, left."
            print(prompt)
            #for event in replicate.stream(
            #        "meta/llama-2-7b-chat",
            #        input={
            #            "top_k": 0,
            #            "top_p": 1,
            #            "prompt": prompt,
            #            "temperature": 0.75,
            #            "system_prompt": "You are the person responsible for determining the direction an agent should move, based on the target position and the agent's current position.You must reach the target. Please respond with a single word: 'up', 'down', 'right', or 'left', without any punctuation or additional explanations.The given coordinates are in the format (x, y), where the x-coordinate increases as you move to the right, and the y-coordinate increases as you move upward.",
            #            "length_penalty": 1,
            #            "max_new_tokens": 8,
            #            "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            #            "presence_penalty": 0
            #        },
            #):
            # for event in replicate.stream(
            #         "meta/meta-llama-3-8b",
            #         input={
            #             "top_k": 0,
            #             "top_p": 0.9,
            #             "prompt": f"You are the person responsible for determining the direction an agent should move, based on the target position and the agent's current position. You must reach your target. The only valid responses you can provide are the following four directions:'Right', 'Left', 'Down', 'Up'.And dont speak sentence, dont use comma and period\n\n{prompt}",
            #             "temperature": 0.5,
            #             "length_penalty": 1,
            #             "max_new_tokens": 5,
            #             "prompt_template": "{prompt}",
            #             "presence_penalty": 1.15
            #         },
            # ):
            #
            #     direction = event.data.strip().lower()
            #     print("Direction:", direction)
            #     if direction:  # 空でない場合のみ処理を行う
            #         if 'up' in direction and self.positions[i][1] < self.wall_y:
            #             self.positions[i][1] = min(self.positions[i][1] + self.dt, self.wall_y)
            #         elif 'down' in direction and self.positions[i][1] > 0:
            #             self.positions[i][1] = max(self.positions[i][1] - self.dt, 0)
            #         elif 'left' in direction and self.positions[i][0] > 0:
            #             self.positions[i][0] = max(self.positions[i][0] - self.dt, 0)
            #         elif 'right' in direction and self.positions[i][0] < self.wall_x:
            #             self.positions[i][0] = min(self.positions[i][0] + self.dt, self.wall_x)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"You are the person responsible for determining the direction an agent should move, based on the target position and the agent's current position.You must reach the target. Please respond with a single word: 'up', 'down', 'right', or 'left', without any punctuation or additional explanations.The given coordinates are in the format (x, y), where the x-coordinate increases as you move to the right, and the y-coordinate increases as you move upward. {prompt}",
                    }
                ],
                model="mixtral-8x7b-32768",
            )
            print(chat_completion.choices[0].message.content)
            direction = chat_completion.choices[0].message.content
            if direction:  # 空でない場合のみ処理を行う
                if 'up' in direction and self.positions[i][1] < self.wall_y:
                    self.positions[i][1] = min(self.positions[i][1] + self.dt, self.wall_y)
                elif 'down' in direction and self.positions[i][1] > 0:
                    self.positions[i][1] = max(self.positions[i][1] - self.dt, 0)
                elif 'left' in direction and self.positions[i][0] > 0:
                    self.positions[i][0] = max(self.positions[i][0] - self.dt, 0)
                elif 'right' in direction and self.positions[i][0] < self.wall_x:
                    self.positions[i][0] = min(self.positions[i][0] + self.dt, self.wall_x)

    def simulate(self):
        plt.ion()
        fig, ax = plt.subplots()
        while True:
            # self.__start_paint()
            self.update_position_based_on_prompt()
            ax.clear()
            ax.set_xlim(0, self.wall_x)
            ax.set_ylim(0, self.wall_y)
            # Draw agents
            ax.scatter(self.positions[:, 0], self.positions[:, 1], color='blue')
            # Draw target with a different color
            ax.scatter(self.target[0], self.target[1], color='red', s=100)  # Adjust size as needed
            plt.pause(0.05)
