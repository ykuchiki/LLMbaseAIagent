import argparse
import sys
import codecs
import time

import torch
import torch.nn as nn
from PIL import Image
from transformers import(
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from transformers import TextStreamer

filename = "time.txt"
def tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    """
    NLPモデルが画像データも認識できるようにするための前処理
    """
    start_time = time.time()
    with codecs.open("check.txt", "w", "utf-8") as f:
        f.write("prompt\n")
        f.write(prompt)
    prompt_chunks = prompt.split("<image>")  # おそらくテキスト内のどこに画像データがあるかのマーカー
    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
    input_ids = tokenized_chunks[0]

    for chunk in tokenized_chunks[1:]:
        input_ids.append(image_token_index)  # 通常のトークンと画像のトークンを区別するために特異的な負のindexを使う
        input_ids.extend(chunk[1:])

    with codecs.open("check.txt", "a", "utf-8") as f:
        f.write("input_ids\n")
        f.write(str(input_ids))
        f.write("torch.tensor(input_ids, dtype=torch.long)\n")
        f.write(str(torch.tensor(input_ids, dtype=torch.long)))
    end_time = time.time()  # 計測終了
    with open(filename, "w") as f:
        print(f"tokenizer_image_token function took {end_time - start_time} seconds", file=f)
    return torch.tensor(input_ids, dtype=torch.long)


def process_tensor(input_ids, image_features, embedding_layer):
    """
    テキストデータと画像データを組み合わせて処理するための前処理
    :param input_ids: テキストを機会処理可能な数値のシーケンスに変換したもの
    :param image_features:
    :param embedding_layer:
    :return:
    """
    start_time = time.time()
    # input_idsの中でindex値が-200の箇所を探す，かつ一番最初に見つかった場所のindexを探す
    split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]

    # -200をのぞいて，input_idsを見つかったインデックスで分割する，-200自体はどちらにも含まれない
    input_ids_1 = input_ids[:, :split_index]
    input_ids_2 = input_ids[:, split_index + 1:]

    # input_idsを埋め込み(ベクトル表現)に変換
    embeddings_1 = embedding_layer(input_ids_1)
    embeddings_2 = embedding_layer(input_ids_2)

    device = image_features.device
    token_embeddings_part1 = embeddings_1.to(device)
    token_embeddings_part2 = embeddings_2.to(device)

    # トークン埋め込みと画像特徴を列方向に結合する
    concatenated_embeddings = torch.cat(
        [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
    )
    with codecs.open("check.txt", "a", "utf-8") as f:
        f.write("concatenated_embedding -> [token_embeddings_part1, image_features, token_embeddings_part2]\n")
        f.write(str(concatenated_embeddings))

    # アテンションマスクの作成，全てのトークンに注目してる．．．
    attention_mask = torch.ones(
        concatenated_embeddings.shape[:2], dtype=torch.long, device=device
    )
    with codecs.open("check.txt", "a", "utf-8") as f:
        f.write("attention_mask\n")
        f.write(str(attention_mask))
    end_time = time.time()  # 計測終了
    with open(filename, "a") as f:
        print(f"process_tensor function took {end_time - start_time} seconds", file=f)
    return concatenated_embeddings, attention_mask


def initialize_models():
    """
    言語モデルと画像認識モデルの初期化と準備
    """
    start_time = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cuda"

    # モデルの量子化設定(数値表現を4bitに圧縮)し，メモリ消費を抑えて計算効率を向上させる
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,   # 量子化をとりあえず無効にする
        bnb_4bit_compute_dtype=torch.float16
    )

    # 事前訓練済みのトークナイザーをロード, use_fast=TrueでRust実装を使用，トークナイゼーションの速度を向上させる
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/llama-3-8b-Instruct", use_fast=True
    )

    model = LlamaForCausalLM.from_pretrained(
        "unsloth/llama-3-8b-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        # quantization_config=bnb_config,
    )

    for param in model.base_model.parameters():
        param.requires_grad = False  # ファインニューニング中にパラメータが更新されないように設定

    model_name = "google/siglip-so400m-patch14-384"
    vision_model = SiglipVisionModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    processor = SiglipImageProcessor.from_pretrained(model_name)  # 画像データの前処理するやつ
    end_time = time.time()  # 計測終了
    with open(filename, "a") as f:
        print(f"initialize_models function took {end_time - start_time} seconds", file=f)

    return tokenizer, model, vision_model, processor


class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),  # ReLUににてるが，ゼロを中心に滑か
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)


def load_projection_module(mm_hidden_size=1152, hidden_size=4096, device="mps"):
    """
    学習済みモデルのロード
    """
    start_time = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    projection_module = ProjectionModule(mm_hidden_size, hidden_size)
    checkpoint = torch.load("./mm_projector.bin")
    checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}  # 辞書型のキーの名前書き換えてる
    projection_module.load_state_dict(checkpoint)
    projection_module = projection_module.to(device).half()
    end_time = time.time()  # 計測終了
    with open(filename, "a") as f:
        print(f"load_projection_module function took {end_time - start_time} seconds", file=f)
    return projection_module

def create_prompt():
    return(
        "You are responsible for determining the direction an agent should move based on its current position and the target position. Follow these guidelines:\n"
        "- The agent must reach the target.\n"
        "- Respond with a single word: 'up', 'down', 'right', or 'left'. Do not use punctuation or extra explanations.\n"
        "- The agent's color is blue, the target's color is red, and obstacles' color is green.\n"
        "Choose one word: 'up', 'down', 'left', or 'right'."
    )



def answer_question(
        image_path, tokenizer, model, vision_model, processor, projection_module
):

    device = "mps" if torch.backends.mps.is_available() else "cuda"
    image = Image.open(image_path).convert("RGB")  # 画像をRGBで読み込む
    tokenizer.eos_token = "<|eot_id|>"  # テキストデータの終わりを示すためのEnd of Sequenceトークンの設定
    prompt_q = create_prompt()

    try:
        start_time = time.time()
        # q = input("\nuser: ")  # ユーザーの入力を受け取る
        q = prompt_q
    except EOFError:
        q = ""
    if not q:
        print("no input detected. exiting.")
        sys.exit()

    question = "<image>" + q
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}" \
             f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    input_ids = (
        tokenizer_image_token(prompt, tokenizer).unsqueeze(0).to(device)
    )

    # TextStreamerはトークン化されたテキストを生成する
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        image_inputs = processor(
            images=[image],
            return_tensors="pt",
            do_resize=True,
            size={"height": 384, "width": 384}
        ).to(device)

        image_inputs = image_inputs["pixel_values"].squeeze(0)

        image_forward_outs = vision_model(
            image_inputs.to(device=device, dtype=torch.float16).unsqueeze(0),
            output_hidden_states=True,
        )

        image_features = image_forward_outs.hidden_states[-2]
        projected_embeddings = projection_module(image_features).to(device)

        embedding_layer = model.get_input_embeddings()

        new_embeds, attn_mask = process_tensor(
            input_ids, projected_embeddings, embedding_layer
        )
        # device = model.device
        attn_mask = attn_mask.to(device)
        new_embeds = new_embeds.to(device)

        model_kwargs = {
            "do_sample": True,
            "temperature": 0.2,
            "max_new_tokens": 10,
            "use_cache": True,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id
        }

        while True:
            print("assistant: ")
            generated_ids = model.generate(
                inputs_embeds=new_embeds, attention_mask=attn_mask, **model_kwargs
            )[0]

            generated_text = tokenizer.decode(generated_ids, skip_tokens=False)
            end_time = time.time()
            with open(filename, "a") as f:
                print(f"LLM generation took {end_time - start_time} seconds", file=f)

            try:
                start_time = time.time()
                # q = input("\nuser: ")
                q = prompt_q
            except EOFError:
                q = ""
            if not q:
                print("no input detected. exiting.")
                # sys.exit()

            new_text = (
                    generated_text
                    + "<|start_header_id|>user<|end_header_id|>\n\n"
                    + q
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            new_input_ids = tokenizer(new_text, return_tensors="pt").input_ids.to(device)
            new_embeddings = embedding_layer(new_input_ids)

            new_embeds =torch.cat([new_embeds, new_embeddings], dim=1)
            attn_mask = torch.ones(new_embeds.shape[:2], device=device)

            end_time = time.time()
            with open(filename, "a") as f:
                print(f"load_projection_module function took {end_time - start_time} seconds", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer question based on an image")
    parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    tokenizer, model, vision_model, processor = initialize_models()
    projection_module = load_projection_module()

    answer_question(
        args.image,
        tokenizer,
        model,
        vision_model,
        processor,
        projection_module,
    )
