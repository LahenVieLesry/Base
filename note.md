1. 读取原始音频文件
wav = librosa.load(wav)

2. 数据增强操作
tri_wav = process(wav)

3. 初始化掩码
mask = np.random

4. 合并两段音频
res_wev = mask * tri_wav + (1 - mask) * wav

5. 转为tensor，使用loss更新
mask = torch.tensor(mask)
upgrade(mask)