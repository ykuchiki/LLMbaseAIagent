import argparse
import sys
import codecs
import time
import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
    TextStreamer
)
filename = "time.txt"
checkfile = "check.txt"

class GetOutput:
    def __init__(self, prompt):
        self.prompt = prompt
        self.tokenizer, self.model, self.vision_model, self.processor = self.initialize_models()
        self.projection_module = self.load_projection_module()

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=-200):
        """
        NLPモデルが画像データも認識できるようにするための前処理
        """
        start_time = time.time()
        with codecs.open(checkfile, "w", "utf-8") as f:
            f.write("prompt文\n")
            f.write(prompt)
            f.write("\n")
        prompt_chunks = prompt.split("<image>")  # おそらくテキスト内のどこに画像データがあるかのマーカー
        tokenized_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        input_ids = tokenized_chunks[0]

        for chunk in tokenized_chunks[1:]:
            input_ids.append(image_token_index)  # 通常のトークンと画像のトークンを区別するために特異的な負のindexを使う
            input_ids.extend(chunk[1:])

        with codecs.open(checkfile, "a", "utf-8") as f:
            f.write("input_ids\n")
            f.write(str(input_ids))
            f.write("\n")
            f.write("torch.tensor(input_ids, dtype=torch.long)\n")
            f.write(str(torch.tensor(input_ids, dtype=torch.long)))
            f.write("\n")
        end_time = time.time()  # 計測終了
        with open(filename, "w") as f:
            print(f"tokenizer_image_token function took {end_time - start_time} seconds", file=f)
        return torch.tensor(input_ids, dtype=torch.long)

    def process_tensor(self, input_ids, image_features, embedding_layer):
        """
        テキストデータと画像データを組み合わせて処理するための前処理
        :param input_ids: テキストを機械処理可能な数値のシーケンスに変換したもの
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
        with codecs.open(checkfile, "a", "utf-8") as f:
            f.write("input_ids_1(テキストを機械処理可能な数値シーケンスに変換したもの)\n")
            f.write(str(input_ids_1))
            f.write("\n")
            f.write("input_ids_2(テキストを機械処理可能な数値シーケンスに変換したもの)\n")
            f.write(str(input_ids_2))
            f.write("\n")

        # input_idsを埋め込み(ベクトル表現)に変換
        embeddings_1 = embedding_layer(input_ids_1)
        embeddings_2 = embedding_layer(input_ids_2)
        with codecs.open(checkfile, "a", "utf-8") as f:
            f.write("embeddings_1(input_ids_1を埋め込みに変換したもの)\n")
            f.write(str(embeddings_1))
            f.write("\n")
            f.write("embeddings_2(input_ids_2を埋め込みに変換したもの)\n")
            f.write(str(embeddings_2))
            f.write("\n")

        device = image_features.device
        token_embeddings_part1 = embeddings_1.to(device)
        token_embeddings_part2 = embeddings_2.to(device)

        # トークン埋め込みと画像特徴を列方向に結合する
        concatenated_embeddings = torch.cat(
            [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
        )
        with codecs.open(checkfile, "a", "utf-8") as f:
            f.write("concatenated_embedding -> [token_embeddings_part1, image_features, token_embeddings_part2]\n")
            f.write(str(concatenated_embeddings))
            f.write("\n")

        # アテンションマスクの作成，全てのトークンに注目してる．．．
        attention_mask = torch.ones(
            concatenated_embeddings.shape[:2], dtype=torch.long, device=device
        )
        with codecs.open(checkfile, "a", "utf-8") as f:
            f.write("attention_mask\n")
            f.write(str(attention_mask))
            f.write("\n")
        end_time = time.time()  # 計測終了
        with open(filename, "a") as f:
            print(f"process_tensor function took {end_time - start_time} seconds", file=f)
        return concatenated_embeddings, attention_mask

    def initialize_models(self):
        """
        言語モデルと画像認識モデルの初期化と準備
        """
        start_time = time.time()
        device = "mps" if torch.backends.mps.is_available() else "cuda"

        print("Initializing models...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_compute_dtype=torch.float16
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/llama-3-8b-Instruct", use_fast=True
        )

        print("Loading model...")
        model = None
        try:
            model = LlamaForCausalLM.from_pretrained(
                "unsloth/llama-3-8b-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                # quantization_config=bnb_config,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

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

    def load_projection_module(self, mm_hidden_size=1152, hidden_size=4096, device="mps"):
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

    def answer_question(self, image_path):
        device = "mps" if torch.backends.mps.is_available() else "cuda"
        image = Image.open(image_path).convert("RGB")  # 画像をRGBで読み込む
        self.tokenizer.eos_token = "<|eot_id|>"  # テキストデータの終わりを示すためのEnd of Sequenceトークンの設定
        t_start_time = time.time()
        q = self.prompt

        # try:
        #     start_time = time.time()
        #     # q = input("\nuser: ")  # ユーザーの入力を受け取る
        #     q = prompt_q
        # except EOFError:
        #     q = ""
        # if not q:
        #     print("no input detected. exiting.")
        #     sys.exit()

        question = "<image>" + q
        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}" \
                 f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer).unsqueeze(0).to(device)
        )

        # TextStreamerはトークン化されたテキストを生成する
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            image_inputs = self.processor(
                images=[image],
                return_tensors="pt",
                do_resize=True,
                size={"height": 384, "width": 384}
            ).to(device)

            image_inputs = image_inputs["pixel_values"].squeeze(0)
            with codecs.open(checkfile, "a", "utf-8") as f:
                f.write(
                    "image_inputs　画像データを前処理して画像認識モデルに入力できる形式に変換したもの，processorインスタンスにより生成\n")
                f.write(str(image_inputs))
                f.write("\n")

            image_forward_outs = self.vision_model(
                image_inputs.to(device=device, dtype=torch.float16).unsqueeze(0),
                output_hidden_states=True,
            )
            with codecs.open(checkfile, "a", "utf-8") as f:
                f.write("image_forward_outs 認識モデルに画像データを入力した後の出力\n")
                f.write(str(image_forward_outs))
                f.write("\n")

            image_features = image_forward_outs.hidden_states[-2]
            with codecs.open(checkfile, "a", "utf-8") as f:
                f.write("image_features image_forward_outsの後ろから2層目の出力だけを抽出したもの\n")
                f.write(str(image_features))
                f.write("\n")
            projected_embeddings = self.projection_module(image_features).to(device)
            with codecs.open(checkfile, "a", "utf-8") as f:
                f.write(
                    "projected_embeddings 画像の特徴ベクトル(image_features)を別の埋め込み空間にマッピングし，テキストの埋め込みベクトルと結合できるようにする\n")
                f.write(str(projected_embeddings))
                f.write("\n")

            embedding_layer = self.model.get_input_embeddings()

            new_embeds, attn_mask = self.process_tensor(
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
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory_summary(device=None, abbreviated=False)

            print("assistant: ")
            generated_ids = self.model.generate(
                inputs_embeds=new_embeds, attention_mask=attn_mask, **model_kwargs
            )[0]

            generated_text = self.tokenizer.decode(generated_ids, skip_tokens=False)
            end_time = time.time()
            with open(filename, "a") as f:
                print(f"LLM generation took {end_time - t_start_time} seconds", file=f)

            return generated_text.strip()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get output based on an image and prompt")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt text")
    parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    output_module = GetOutput(prompt=args.prompt, env=None)
    result = output_module.answer_question(image_path=args.image)
    print(f"Result: {result}")
