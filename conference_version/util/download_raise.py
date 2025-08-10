import os
import requests
from tqdm import tqdm

# 指定下载文件夹路径
download_folder = 'data/RAISE'
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# 读取CSV文件
with open('util/RAISE_1k.csv', 'r') as file:
    lines = file.readlines()

# 遍历每一行（从第二行开始）
for line in tqdm(lines[1:], desc="Downloading", unit="file"):
    # 解析CSV行数据
    data = line.strip().split(',')

    # 获取TIFF文件链接
    tiff_url = data[2]

    # 提取文件名
    filename = tiff_url.split('/')[-1]

    # 构建文件的完整路径
    filepath = os.path.join(download_folder, filename)

    # 发起请求并下载TIFF文件
    response = requests.get(tiff_url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        tqdm.write(f"下载成功：{filename}")
    else:
        tqdm.write(f"下载失败：{filename}")