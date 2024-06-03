import numpy as np
import matplotlib.pyplot as plt
from groq import Groq

from apikey import GROG_API_KEY

from multimodal.llava import Llava
# from multimodal.llamasiglip import Llama3_SigLip

client = Groq(
    api_key=GROG_API_KEY,
)


class SimulateLLMAgent:
    def __init__(self, people_num, wall_x, wall_y, dt, obstacle_num=3, log_length=3):
        print("Initializing SimulateLLMAgent...")
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
        self.llm_model = Llava()
        self.log_file = "log_simulation.txt"
        self.replay_file = "log_replay.txt"
        self.log_length = log_length * 2
        self.logs = []
        self.log_conversation = "log_conversation.txt"
        self.log_prompt = "log_prompt.txt"
        self.no_change_counter = {i: 0 for i in range(people_num)}  # 位置が変わらなかった回数をカウント
        self.previous_direction = {i: None for i in range(people_num)}  # 各エージェントの前回の方向を記録

        # log_conversation.txtの中身を空にする
        with open(self.log_conversation, 'w') as log_file:
            log_file.write('')

        print("SimulateLLMAgent initialized.")

    def log(self, filename, message):
        with open(filename, "a") as log_file:
            log_file.write(message + "\n")

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

    def is_near_obstacle(self, position, obstacle, radius):
        start_point, end_point = obstacle
        return self.point_to_line_distance(position, start_point, end_point) <= radius

    def create_prompt(self, i):
        log_text = "Previous interactions:\n" + "\n".join(self.logs)
        return (
            "You are responsible for determining the direction an agent should move based on its current position and the target position. Follow these guidelines:\n"
            "- The agent must reach the target.\n"
            "- Respond with a single word: 'up', 'down', 'right', or 'left'. Do not use punctuation or extra explanations.\n"
            "- Avoid any obstacles in the nearby area.\n"
            "- The agent's color is blue, the target's color is red, and obstacles' color is green.\n"
            "- Don't get too close to obstacles\n"
            "- Path Planning: Consider the obstacles in the path and choose the direction that avoids them while still progressing towards the target.\n"
            f"Current position: (x, y) = ({self.positions[i][0]:.2f}, {self.positions[i][1]:.2f})\n"
            f"Target position: (x, y) = ({self.target[0]:.2f}, {self.target[1]:.2f})\n"
            "Choose one word: 'up', 'down', 'left', or 'right'."
        )

    def update_position_based_on_prompt(self, ax):
        original_size = 100  # 元のエージェントのサイズ
        extraction_size = 100  # 抽出する範囲のサイズ（30の半径の両側）
        scale_factor = self.wall_x / extraction_size  # 比率計算
        adjusted_size = original_size * scale_factor  # サイズを比率に基づいて調整
        for i in range(self.people_num):
            if self.control_mode == 'manual' and i == self.user_controlled_agent:
                direction = self.direction
            else:
                prompt = self.create_prompt(i)
                previous_logs = self.load_logs()
                complete_prompt = f"Previous Logs:\n {previous_logs}\nCurrent Question:\n{prompt}"

                # ============================================================
                # GrogAPI使う時(テキストのみ)
                # chat_completion = client.chat.completions.create(
                #     messages=[
                #         {
                #             "role": "user",
                #             "content": prompt
                #         }
                #     ],
                #     model="llama3-70b-8192",
                # )
                # print("prompt: ", prompt)
                # direction = chat_completion.choices[0].message.content
                # ============================================================

                # シミュレーションの状態を画像として保存
                agent_position = self.positions[i]
                ax.clear()

                # 境界を考慮して範囲を調整
                x_min = max(agent_position[0] - 50, 0)
                x_max = min(agent_position[0] + 50, self.wall_x)
                y_min = max(agent_position[1] - 50, 0)
                y_max = min(agent_position[1] + 50, self.wall_y)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.scatter(self.positions[:, 0], self.positions[:, 1], color='blue', s=adjusted_size)
                ax.scatter(self.target[0], self.target[1], color='red', s=adjusted_size * 2)
                for start, end in self.obstacles:
                    ax.plot([start[0], end[0]], [start[1], end[1]], color="green")
                plt.savefig("current_state.png")
                # ============================================================
                # Hugging faceのLlama3とSigLip，事前学習済みProjection Layerを用いたマルチモーダル
                # direction = self.get_output.answer_question(image_path="current_state.png", prompt=prompt)
                # reason = self.get_output.answer_question(None, direction)
                # ============================================================

                # ============================================================
                # Llavaによるテキスト質問と画像入力のマルチモーダル
                direction = self.llm_model.get_output(prompt=complete_prompt, image="current_state.png")
                # ============================================================

                self.add_log(f"user: {prompt}\n")
                self.add_log(f"AI(You): {direction}\n\n\n\n")
                # print("wow", direction)

            if direction:  # 空でない場合のみ処理を行う
                original_position = self.positions[i].copy()
                if ('up' in direction or 'Up' in direction) and self.positions[i][1] < self.wall_y:
                    self.positions[i][1] = min(self.positions[i][1] + self.dt, self.wall_y)
                elif ('down' in direction or 'Down' in direction) and self.positions[i][1] > 0:
                    self.positions[i][1] = max(self.positions[i][1] - self.dt, 0)
                elif ('left' in direction or 'Left' in direction) and self.positions[i][0] > 0:
                    self.positions[i][0] = max(self.positions[i][0] - self.dt, 0)
                elif ('right' in direction or 'Right' in direction) and self.positions[i][0] < self.wall_x:
                    self.positions[i][0] = min(self.positions[i][0] + self.dt, self.wall_x)

                # 衝突チェック
                if any(self.point_to_polygon_distance(self.positions[i], np.array(obs)) < 3 for obs in self.obstacles):
                    self.positions[i] = original_position

                # ログに位置を記録
                self.log(self.replay_file, f"Agent {i} position: {self.positions[i]}")

                # 複数回エージェントが移動できなかった時
                if np.array_equal(self.positions[i], original_position) and self.control_mode == "LLM":
                    print("no change counter:", self.no_change_counter)
                    self.no_change_counter[i] += 1
                    if self.no_change_counter[i] >= 2:  # n回の座標移動なし
                        previous_direction = self.previous_direction[i]
                        reason_prompt = (
                            f"The agent's position has not changed for {self.no_change_counter[i]} updates.\n"
                            f"Previous direction: {previous_direction}.\n"
                            "Why didn't the position change? Respond with a reason."
                        )
                        previous_logs = self.load_logs()
                        complete_prompt = f"Previous Logs:\n {previous_logs}\nCurrent Question:\n{reason_prompt}"
                        # ============================================================
                        # 理由を聞く仕組みをここに作る
                        reason = self.llm_model.get_output(reason_prompt, image="current_state.png")
                        # ============================================================

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
        self.log(self.replay_file, f"Initial positions: {self.positions}")
        self.log(self.replay_file, f"Target position: {self.target}")
        self.log(self.replay_file, f"Obstacles: {self.obstacles}")
        while not self.exit_simulation:
            # self.__start_paint()
            self.update_position_based_on_prompt(ax)
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

    def replay(self):
        with open(self.replay_file, "r") as log_file:
            log_data = log_file.readlines()

        # eval() : 文字列をnumpyに変換
        # .split(":")[1] : 指定した文字列で分割して二番目の要素を取得
        # .strip() : 空白を取り除く
        initial_positions = eval(log_data[0].split(":")[1].strip())
        target_position = eval(log_data[1].split(":")[1].strip())
        obstacles = eval(log_data[2].split(":")[1].strip())

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.wall_x)
        ax.set_ylim(0, self.wall_y)
        ax.scatter(initial_positions[:, 0], initial_positions[:, 1], color='blue')
        ax.scatter(target_position[0], target_position[1], color='red', s=100)
        for start, end in obstacles:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="green")

        for line in log_data[3:]:
            if line.startswith("Agent"):
                agent_index = int(line.split(" ")[1])
                position = eval(line.split(":")[1].strip())
                ax.scatter(position[0], position[1], color='blue')
            plt.pause(0.05)

        plt.show()
