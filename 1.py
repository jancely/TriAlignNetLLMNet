import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 读取数据
# ============================================================

file_path = r"E:\竞赛\Datasets\protein.csv"
output_path = r"E:\竞赛\Datasets\protein_output.csv"

df = pd.read_csv(file_path)

# 日期转换
df["DATE"] = pd.to_datetime(df["DATE"])

# 按时间排序
df = df.sort_values("DATE").reset_index(drop=True)


# ============================================================
# 2. 获取数值变量
# ============================================================

feature_cols = [
    "Broiler_Exports",
    "Broiler_Imports",
    "Egg_product_shell_egg_equivalent_Exports",
    "Egg_product_shell_egg_equivalent_Imports",
    "Egg_total_including_products_shell_egg_equivalent_Exports",
    "Egg_total_including_products_shell_egg_equivalent_Imports",
    "Other_chicken_Exports",
    "Other_chicken_Imports",
    "Shell_egg_Exports",
    "Shell_egg_Imports",
    "Turkey_Exports",
    "Turkey_Imports"
]


# ============================================================
# 3. 保留原始缺失信息
#    NaN      -> 1
#    非 NaN   -> 0
# ============================================================

for col in feature_cols:

    df[col + "_missing"] = df[col].isna().astype(int)


# ============================================================
# 4. 时间线性插值
# ============================================================

for col in feature_cols:

    df[col] = df[col].interpolate(
        method="linear",
        limit_direction="both"
    )


# ============================================================
# 5. 检查是否还有缺失值
# ============================================================

print("插值之后的缺失值：")

print(
    df[feature_cols].isna().sum()
)


# ============================================================
# 6. 查看缺失值处理前后的情况
# ============================================================

for col in feature_cols:

    missing_col = col + "_missing"

    if df[missing_col].sum() > 0:

        print("\n变量：", col)

        print(
            df.loc[
                df[missing_col] == 1,
                ["DATE", col, missing_col]
            ]
        )


# ============================================================
# 7. 构造 MLP 输入变量
# ============================================================

missing_cols = [
    col + "_missing"
    for col in feature_cols
]

X_value = df[feature_cols].values
X_missing = df[missing_cols].values

# 数值变量 + 缺失指示变量
X = np.concatenate(
    [X_value, X_missing],
    axis=1
)
X = pd.DataFrame(
    X,
    columns=feature_cols + missing_cols
)
X.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n原始数值变量维度：", X_value.shape)
print("缺失指示变量维度：", X_missing.shape)
print("最终 MLP 输入维度：", X.shape)


# ============================================================
# 8. StandardScaler 标准化
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 9. 查看最终数据
# ============================================================

print("\n最终输入数据：")
print(X_scaled[:5])

print("\n最终数据维度：")
print(X_scaled.shape)