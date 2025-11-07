import os
from volcenginesdkarkruntime import Ark

# 从环境变量读取API密钥,避免硬编码泄露
# 使用方法: 设置环境变量 DOUBAO_API_KEY
# Linux/Mac: export DOUBAO_API_KEY="your-api-key"
# Windows: set DOUBAO_API_KEY=your-api-key
client = Ark(api_key=os.getenv('DOUBAO_API_KEY', 'YOUR_API_KEY_HERE'))

def target_objects(text: str):
    """
    安全检查：
    1. Ark API 调用失败 → 抛出异常
    2. 未识别到物体名称 → 抛出异常
    3. 识别成功 → 返回物体名称列表（Python list）
    """
    try:
        completion = client.chat.completions.create(
            model="doubao-1.5-pro-32k-250115",
            messages=[
                {"role": "system",
                 "content": (
                     "你现在是一个助老用户命令判断助手。你会收到老人提出的问题，其中包含老人要寻找的物品名称。"
                     "你要做的是提取物品名称。注意，老人的提问可能包含一个或多个物品名称。"
                     "你的回答只能为以下格式：[物品名称1, 物品名称2, ...]。"
                     "如果你无法从问题中提取任何物品名称，请回答：[]。"
                 )},
                {"role": "user", "content": f"文段：{text}"}
            ],
        )
    except Exception as e:
        # Step 1️⃣ Ark API 调用失败
        raise RuntimeError(f"Ark API 调用失败：{e}")

    # Step 2️⃣ 检查 API 是否返回有效内容
    try:
        response_text = completion.choices[0].message.content.strip()
    except Exception:
        raise ValueError("API 返回内容格式错误，无法解析 message.content")

    # Step 3️⃣ 解析输出为 Python 列表
    cleaned = (
        response_text.strip()
        .replace("，", ",")
        .replace(" ", "")
        .strip()
        .strip("[]")
    )

    # 可能为空数组的情况
    if not cleaned:
        raise ValueError("未识别到任何物品名称")

    # 转成列表对象
    objects = [i for i in cleaned.split(",") if i]

    # 如果解析后为空，也抛出异常
    if not objects:
        raise ValueError("未识别到任何物品名称")

    # Step 4️⃣ 成功返回
    return objects
def evaluate_targe_object(target,result_dict):
    completion = client.chat.completions.create(
        model="doubao-1.5-pro-32k-250115",
        messages=[
            {"role": "system",
             "content": (
                 "你会收到两个输入：一个是用户想找的物品名称列表，另一个是算法识别到的物品名称字典。"
                 "你的任务是判断用户想找的物品名称是否在识别到的物品名称中出现。如果出现，返回：识别到目标物体，信息为：{result_dict}"
                 "其中{result_dict}是识别到的物品名称字典内容，要保留对应目标物体的部分"
                 "如果没有出现，返回：未识别到目标物体"
                 "用户可能要多个目标物体，对于每个目标物体都要进行判断，分别输出每个目标物体的识别情况"
                 "注意，用户的目标物体为中文，result_dict为英文。请留意。"
                 
             )},
            {"role": "user", "content": f"目标物体：{target}, 识别到的物体信息：{result_dict}"}
        ],
    )
    return completion.choices[0].message.content