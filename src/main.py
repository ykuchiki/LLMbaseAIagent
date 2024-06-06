import argparse
from simpleSFM import people_flow
from llmagent import people_flow1
from multimodal.multimodalagent import SimulateLLMAgent

def initialize_parameters():
    # シミュレーション設定の初期化
    params = {
        'people_num': 1,
        'v_arg': [6, 2],
        'repul_h': [5, 5],
        'repul_m': [2, 2],
        'target': [[60, 240], [120, 150], [90, 60], [240, 40], [200, 120], [170, 70], [150, 0]],
        'R': 3,
        'min_p': 0.1,
        'p_arg': [[0.5, 0.1]],
        'wall_x': 300,
        'wall_y': 300,
        'in_target_d': 3,
        'dt': 0.1,
        'dt2': 5,
        'save_format': "heat_map",
        'save_params': [(30, 30), 1],
        'obstacle_num': 6
    }
    return params

def main(model_name):
    params = initialize_parameters()

    if model_name == "simpleSFM":
        model = people_flow(params['people_num'], params['v_arg'], params['repul_h'], params['repul_m'], params['target'], params['R'], params['min_p'], params['p_arg'], params['wall_x'], params['wall_y'], params['in_target_d'], params['dt'], save_format=params['save_format'], save_params=params['save_params'])
    elif model_name == "llmagent":
        model = people_flow1(params['people_num'], params['wall_x'], params['wall_y'], params['dt2'], params['obstacle_num'], log_length=15)
    elif model_name == "multimodal":
        model = SimulateLLMAgent(params['people_num'], params['wall_x'], params['wall_y'], params['dt2'], params['obstacle_num'], log_length=3)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # シミュレーションの実行
    # model.simulate()
    model.replay()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the people flow simulation with different models.")
    parser.add_argument("model", choices=["simpleSFM", "llmagent", "multimodal"], help="Choose the model to run the simulation")
    args = parser.parse_args()
    main(args.model)
