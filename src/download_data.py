from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=350)

# gộp đặc trưng và biến mục tiêu thành một bảng
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

out_path = Path("data/raw/default_credit_card_clients.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(df.shape)