import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_dir = r"E:\竞赛\LLMs\GPT4All\Qwen2"
hf_model_name = ("DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")
gguf_path = os.path.join(model_dir, hf_model_name)
#
tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    trust_remote_code=True)

print("正在加载 GGUF 模型...")
llm_model = AutoModelForCausalLM.from_pretrained(
                                                model_dir,
                                                gguf_file=gguf_path,
                                                local_files_only=True,
                                                dtype=torch.float32,
                                                trust_remote_code=True)


text_path = r'E:/竞赛/Datasets/Text/TextTransformed.csv'
text_df = pd.read_csv(text_path,  encoding="utf-8-sig")
text_df['date'] = pd.to_datetime(text_df['col1'])

text_df['month'] = (text_df['date'].dt.to_period('M'))

text_df['text'] = text_df['col2'].fillna('No information available').tolist()

text_flattened = text_df['text'].tolist()

tokenized_output = tokenizer(
                            text_flattened,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=256
                            )
input_ids = tokenized_output['input_ids'].to(llm_model.device)
#
attn_mask = tokenized_output['attention_mask'].to(llm_model.device)
embedding_layer = (llm_model.get_input_embeddings())
#
text_embeddings = embedding_layer(input_ids)
#
#
# # Token-level pooling
# # ============================================================
pool_type = "avg"
#
expanded_mask = (attn_mask.unsqueeze(-1).expand_as(text_embeddings))
#
if pool_type == "avg":
    masked_emb = (text_embeddings * expanded_mask)
#
    valid_counts = (expanded_mask.sum(dim=1).clamp(min=1))
#
    text_embeddings = (masked_emb.sum(dim=1) / valid_counts)
#
# elif pool_type == "max":
#     masked_emb = text_embeddings.masked_fill(
#                                             expanded_mask == 0,
#                                             float('-inf'))
#
#     text_embeddings, _ = masked_emb.max(dim=1)
#
# elif pool_type == "min":
#     masked_emb = text_embeddings.masked_fill(
#                                             expanded_mask == 0,
#                                             float('inf'))
#
#     text_embeddings, _ = masked_emb.min(dim=1)
#
#
# # 周 → 月
# # ============================================================
weekly_embeddings = text_embeddings
monthly_embeddings = []
#
for month in text_df['date'].dt.to_period('M'):
    indices = np.where(text_df['month'].values == month)[0]

    if len(indices) == 0:
        # 没有对应文本
        month_embedding = torch.zeros(
                                        weekly_embeddings.shape[-1],
                                        device=weekly_embeddings.device)

    else:
        month_embedding = (
                            weekly_embeddings[indices]
                            .mean(dim=0))

    monthly_embeddings.append(month_embedding)

monthly_embeddings = torch.stack(monthly_embeddings)

# 按 train / validation / test 划分
# ============================================================

# text_embeddings = (monthly_embeddings[border1:border2])

weekly_embeddings_np = (weekly_embeddings.detach().cpu().numpy())
monthly_embeddings_np = (monthly_embeddings.detach().cpu().numpy())

# 构造 DataFrame
embedding_dim = weekly_embeddings_np.shape[1]
# embedding_dim_month = monthly_embeddings_np.shape[1]

embedding_columns = [
                    f"text_emb_{i}"
                    for i in range(embedding_dim)
                    ]

df_weekly_embedding = pd.DataFrame(
                                    weekly_embeddings_np,
                                    columns=embedding_columns
                                    )
df_monthly_embedding = pd.DataFrame(
                                    monthly_embeddings_np,
                                    columns=embedding_columns
                                    )

# 加上原始文本日期
df_weekly_embedding.insert(
                            0,
                            "date",
                            text_df["date"].values
                            )


# 加上月份
df_weekly_embedding.insert(
                            1,
                            "month",
                            text_df["month"].astype(str).values
                            )
df_monthly_embedding.insert(
                            0,
                            "month",
                            text_df["month"].values
                            )

# 保存
output_path = os.path.join(
                            'E:/竞赛/Datasets/Text',
                            "weekly_text_embeddings.csv"
                            )
output_path2 = os.path.join(
                            'E:/竞赛/Datasets/Text',
                            "monthly_text_embeddings.csv"
                            )

df_weekly_embedding.to_csv(
                            output_path,
                            index=False,
                            encoding="utf-8-sig"
                            )
df_monthly_embedding.to_csv(
                            output_path2,
                            index=False,
                            encoding="utf-8-sig"
                            )

print("Weekly Text Embedding saved to:", output_path)

print("Shape:", df_weekly_embedding.shape)
print("Shape:", df_monthly_embedding.shape)