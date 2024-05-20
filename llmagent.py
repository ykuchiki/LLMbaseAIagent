import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
from apikey import GROG_API_KEY

client = Groq(
    api_key=GROG_API_KEY,
)


class people_flow1:
    def __init__(self, people_num, wall_x, wall_y, dt, obstacle_num=3):
        self.people_num = people_num
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.dt = dt
        self.positions = np.random.rand(people_num, 2) * [wall_x, wall_y]
        self.target = np.array([wall_x / 2, wall_y / 2])
        self.obstacles = self.generate_obstacles(obstacle_num)
        self.user_controlled_agent = 0  # 入力で操作するエージェントのインデックス
        self.direction = None  # ユーザーの入力方向
        self.control_mode = "manual"  # 制御モード LLM or manual
        self.exit_simulation = False

    def generate_random_points(self):
        while True:
            start_point = np.random.rand(2) * [self.wall_x, self.wall_y]
            end_point = np.random.rand(2) * [self.wall_x, self.wall_y]
            distance = np.abs(end_point - start_point).sum()
            if 60 <= distance <= 90:  # マンハッタン距離が3以上5以下の障害物
                return start_point, end_point

    def generate_obstacles(self, obstacle_num):
        obstacle_pairs = []
        for _ in range(obstacle_num):
            start_point, end_point = self.generate_random_points()
            obstacle_pairs.append((start_point, end_point))
        return obstacle_pairs

    def format_obstacles_for_prompt(self):
        formatted_obstacles = []
        for pair in self.obstacles:
            formatted_obstacles.append(f"({pair[0][0]:.2f}, {pair[0][1]:.2f}) to ({pair[1][0]:.2f}, {pair[1][1]:.2f})")
        return "; ".join(formatted_obstacles)

    def create_prompt(self, i):
        obstacles_text = self.format_obstacles_for_prompt()
        prompt = (
            "You are responsible for determining the direction an agent should move based on its current position and the target position. Follow these guidelines:\n"
            "- The agent must reach the target.\n"
            "- Respond with a single word: 'up', 'down', 'right', or 'left'. Do not use punctuation or extra explanations.\n"
            "- Coordinates are given in (x, y) format, where x increases to the right and y increases upward.\n"
            f"Current position: (x, y) = ({self.positions[i][0]}, {self.positions[i][1]})\n"
            f"Target position: (x, y) = ({self.target[0]}, {self.target[1]})\n"
            f"Obstacles are located at: [{obstacles_text}]\n"
            "Choose one word: 'up', 'down', 'left', or 'right'.\n"
        )
        return prompt

    def point_to_line_distance(self, point, line_start, line_end):
        """入力された点と線分の最短距離を計算"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = line_start + t * line_vec
        # print(np.linalg.norm(point - projection))
        return np.linalg.norm(point - projection)

    def update_position_based_on_prompt(self):
        for i in range(self.people_num):
            if self.control_mode == 'manual' and i == self.user_controlled_agent:
                direction = self.direction
            else:
                prompt = self.create_prompt(i)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="mixtral-8x7b-32768",
                )
                # print("prompt: ", prompt)
                direction = chat_completion.choices[0].message.content
            print(direction)
            if direction:  # 空でない場合のみ処理を行う
                original_position = self.positions[i].copy()
                if 'up' in direction and self.positions[i][1] < self.wall_y:
                    self.positions[i][1] = min(self.positions[i][1] + self.dt, self.wall_y)
                elif 'down' in direction and self.positions[i][1] > 0:
                    self.positions[i][1] = max(self.positions[i][1] - self.dt, 0)
                elif 'left' in direction and self.positions[i][0] > 0:
                    self.positions[i][0] = max(self.positions[i][0] - self.dt, 0)
                elif 'right' in direction and self.positions[i][0] < self.wall_x:
                    self.positions[i][0] = min(self.positions[i][0] + self.dt, self.wall_x)

                # 衝突チェック
                if any(self.point_to_line_distance(self.positions[i], obs[0], obs[1]) < 3 for obs in self.obstacles):
                    self.positions[i] = original_position

            if self.control_mode == "manual":
                self.direction = None

    def on_key_press(self, event):
        if event.key == 'up':
            self.direction = 'up'
        elif event.key == 'down':
            self.direction = 'down'
        elif event.key == 'left':
            self.direction = 'left'
        elif event.key == 'right':
            self.direction = 'right'
        elif event.key == 'escape':
            self.exit_simulation = True
        elif event.key == 'm':
            self.control_mode = 'manual'
            print("Control mode: manual")
        elif event.key == 'l':
            self.control_mode = 'LLM'
            print("Control mode: LLM")

    def simulate(self):
        plt.ion()
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        while not self.exit_simulation:
            # self.__start_paint()
            self.update_position_based_on_prompt()
            ax.clear()
            ax.set_xlim(0, self.wall_x)
            ax.set_ylim(0, self.wall_y)
            # Draw agents
            ax.scatter(self.positions[:, 0], self.positions[:, 1], color='blue')
            # Draw target with a different color
            ax.scatter(self.target[0], self.target[1], color='red', s=100)  # Adjust size as needed
            # print(self.obstacles)
            for start, end in self.obstacles:
                ax.plot([start[0], end[0]], [start[1], end[1]], color="green")
            plt.pause(0.05)

            # 終了条件のチェック
            for pos in self.positions:
                if np.linalg.norm(pos - self.target) < self.dt:
                    print("ターゲットに到着しました！")
                    self.exit_simulation = True
                    break

        print("シミュレーションを終了します")
        plt.ioff()
        plt.close()
