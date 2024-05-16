import os
# import replicate

# os.environ["REPLICATE_API_TOKEN"] = "API key"
#
# # The meta/llama-2-7b-chat model can stream output as it's running.
# for event in replicate.stream(
#     "meta/llama-2-7b-chat",
#     input={
#         "top_k": 0,
#         "top_p": 1,
#         "prompt": "agent:(x, y) = (0, 0), target = (0, 1)",
#         "temperature": 0.75,
#         "system_prompt": "You are the person responsible for determining the direction an agent should move, based on the target coordinates and the agent's current position. The only valid responses you can provide are the following four directions:Right, Left, Down, Up.And dont speak sentence, dont use comma and period",
#         "length_penalty": 1,
#         "max_new_tokens": 8,
#         "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
#         "presence_penalty": 0
#     },
# ):
#         print(str(event), end="")

# from groq import Groq
#
# client = Groq(
#     # This is the default and can be omitted
#     api_key="api-key",
# )
#
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "You are the person responsible for determining the direction an agent should move, based on the target position and the agent's current position.You must reach the target. Please respond with a single word: 'up', 'down', 'right', or 'left', without any punctuation or additional explanations.The given coordinates are in the format (x, y), where the x-coordinate increases as you move to the right, and the y-coordinate increases as you move upward. Agent at (17.19, 140.78), target at (150.00, 150.00). Choose: up, down, right, left.",
#         }
#     ],
#     model="mixtral-8x7b-32768",
# )
# print(chat_completion.choices[0].message.content)

import torch
print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())