import glob
import os

files = glob.glob("/home/flux/graph_eqa_swagat/spatial_experiment/multi_agent/agents/openai_*.py")
for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Rename class
    content = content.replace("class OpenAI", "class Alibaba")
    content = content.replace("model_name=\"gpt-5.2-chat-latest\"", "model_name=\"qwen3-max\"")
    content = content.replace("OPENAI_API_KEY", "ALIBABA_API_KEY")
    content = content.replace("(OpenAI)", "(Alibaba)")

    # Client instantiation
    old_client_code = 'self.client = OpenAI(api_key=os.environ["ALIBABA_API_KEY"])'
    new_client_code = 'self.client = OpenAI(\n            api_key=os.environ["ALIBABA_API_KEY"],\n            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1"\n        )'
    content = content.replace(old_client_code, new_client_code)

    new_file = file.replace("openai_", "alibaba_")
    with open(new_file, "w") as f:
        f.write(content)
    print(f"Created {new_file}")

