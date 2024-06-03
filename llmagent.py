import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
from apikey import GROG_API_KEY

client = Groq(
    api_key=GROG_API_KEY,
)


class people_flow1:
    def __init__(self, people_num, wall_x, wall_y, dt, obstacle_num=3, log_length=3):
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
        self.log_length = log_length * 2
        self.logs = []
        self.log_conversation = "log_conversation.txt"
        self.log_prompt = "log_prompt.txt"
        self.no_change_counter = {i: 0 for i in range(people_num)}  # 位置が変わらなかった回数をカウント
        self.previous_direction = {i: None for i in range(people_num)}  # 各エージェントの前回の方向を記録

    def load_logs(self):
        try:
            with open(self.log_conversation, "r") as log_file:
                logs = log_file.read()
                return logs
        except FileNotFoundError:
            return ""

    def save_prompt(self, prompt):
        with open(self.log_prompt, "w") as log_file:
            log_file.write("".join(prompt))

    def add_log(self, log_entry):
        # デバッグ出力
        # print(f"Attempting to log: {log_entry}")

        if len(self.logs) >= self.log_length:
            self.logs.pop(0)
        self.logs.append(log_entry)

        with open(self.log_conversation, "w") as log_file:
            log_file.write("".join(self.logs))

    def generate_random_points(self):
        while True:
            start_point = np.random.rand(2) * [self.wall_x, self.wall_y]
            end_point = np.random.rand(2) * [self.wall_x, self.wall_y]
            distance = np.abs(end_point - start_point).sum()
            if 60 <= distance <= 90:  # マンハッタン距離が60以上90以下の障害物
                return start_point, end_point

    def generate_obstacles(self, obstacle_num):
        obstacle_pairs = []
        for _ in range(obstacle_num):
            start_point, end_point = self.generate_random_points()
            obstacle_pairs.append((start_point, end_point))
        return obstacle_pairs

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

    def point_to_polygon_distance(self, point, polygon):
        """点とポリゴンの最小の距離を計算"""
        min_distance = float("inf")
        for i in range(len(polygon)):
            line_start = polygon[i]
            line_end = polygon[(i + 1) % len(polygon)]
            distance = self.point_to_line_distance(point, line_start, line_end)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def format_obstacles_for_prompt(self, i, radius=30):
        position = self.positions[i]
        nearby_obstacles = self.get_nearby_obstacles(position, radius)
        formatted_obstacles = []
        for obstacle in nearby_obstacles:
            start_point, end_point = obstacle
            formatted_obstacles.append(
                f"(({start_point[0]:.2f}, {start_point[1]:.2f}), ({end_point[0]:.2f}, {end_point[1]:.2f}))"
            )
        return formatted_obstacles

    def get_nearby_obstacles(self, position, radius=30):
        nearby_obstacles = []
        for obstacle in self.obstacles:
            if self.is_near_obstacle(position, obstacle, radius):
                nearby_obstacles.append(obstacle)
        return nearby_obstacles

    def is_near_obstacle(self, position, obstacle, radius):
        start_point, end_point = obstacle
        return self.point_to_line_distance(position, start_point, end_point) <= radius

    def create_prompt(self, i):
        obstacles_text = self.format_obstacles_for_prompt(i)
        obstacles_list = ', '.join(obstacles_text)
        prompt = (
            "Instruction:\n" 
            "You are responsible for determining the direction an agent should move based on its current position and the target position. Follow these guidelines:\n"
            "- The agent must reach the target.\n"
            "- Respond with a single word: 'up', 'down', 'right', or 'left'. Do not use punctuation or extra explanations.\n"
            "- The given coordinates are in the format (x, y), where the x-coordinate increases as you move to the right, and the y-coordinate increases as you move upward. \n"
            "- Avoid any obstacles in the nearby area.\n"
            f"Current position: (x, y) = ({self.positions[i][0]:.2f}, {self.positions[i][1]:.2f})\n"
            f"Target position: (x, y) = ({self.target[0]:.2f}, {self.target[1]:.2f})\n"
            f"Nearby obstacles (x, y): {obstacles_list}\n"
            "Choose one word: 'up', 'down', 'left', or 'right'.\n"
            )
        return prompt, obstacles_list

    def update_position_based_on_prompt(self):
        for i in range(self.people_num):
            if self.control_mode == 'manual' and i == self.user_controlled_agent:
                direction = self.direction
            else:
                prompt, obstacles_list = self.create_prompt(i)
                previous_logs = self.load_logs()
                complete_prompt = f"Previous Logs:\n {previous_logs}\nCurrent Question:\n{prompt}"
                self.save_prompt(complete_prompt)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": complete_prompt
                        }
                    ],
                    model="llama3-70b-8192",
                )
                # print("prompt: ", prompt)
                direction = chat_completion.choices[0].message.content
                self.add_log(f"user: {prompt}\n")
                self.add_log(f"AI(You): {direction}\n\n\n\n")

                # chat_completion1 = client.chat.completions.create(
                #     messages=[
                #         {
                #             "role": "user",
                #             "content": f"Why did you choose the {direction}. Answer the reason why."
                #         }
                #     ],
                #     model="llama3-8b-8192",
                # )
                # reason = chat_completion1.choices[0].message.content
                # print(reason)

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

                # 衝突判定
                if any(self.point_to_polygon_distance(self.positions[i], np.array(obs)) < 3 for obs in self.obstacles):
                    self.positions[i] = original_position

                if np.array_equal(self.positions[i], original_position) and self.control_mode == "LLM":
                    print("no change counter:", self.no_change_counter)
                    self.no_change_counter[i] += 1
                    if self.no_change_counter[i] >= 2:  # 座標が指定回数以上変化がないとき
                        previous_direction = self.previous_direction[i]
                        reason_prompt = (
                            f"The agent's position has not changed for {self.no_change_counter[i]} updates.\n"
                            f"Current position: ({self.positions[i][0]:.2f}, {self.positions[i][1]:.2f}).\n"
                            f"Previous direction: {previous_direction}.\n"
                            f"Nearby obstacles (x, y): {obstacles_list}\n"
                            "Why didn't the position change? Respond with a reason."
                        )
                        reason_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": reason_prompt
                                }
                            ],
                            model="llama3-70b-8192",
                        )
                        reason = reason_completion.choices[0].message.content
                        self.add_log(f"user: {reason_prompt}\n")
                        self.add_log(f"AI(You): {reason}\n\n\n\n")
                        self.no_change_counter[i] = 0  # カウンタをリセット

                self.previous_direction[i] = direction  # 前回の方向を記録

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

            #fig.canvas.draw_idle()
            #fig.canvas.flush_events()

        print("シミュレーションを終了します")
        plt.ioff()
        plt.close()
