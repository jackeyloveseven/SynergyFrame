import os

src_dir = os.path.join(os.path.dirname(__file__), 'src_image')
tgt_dir = os.path.join(os.path.dirname(__file__), 'tgt_image')

# 获取并排序文件名
src_files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])
tgt_files = sorted([f for f in os.listdir(tgt_dir) if os.path.isfile(os.path.join(tgt_dir, f))])

# 检查数量是否一致
if len(src_files) != len(tgt_files):
    print("源文件和目标文件数量不一致，无法一一对应重命名！")
    exit(1)

for src_name, tgt_name in zip(src_files, tgt_files):
    tgt_path = os.path.join(tgt_dir, tgt_name)
    new_tgt_path = os.path.join(tgt_dir, src_name)
    # 如果目标名已存在，先删除
    if tgt_path != new_tgt_path:
        if os.path.exists(new_tgt_path):
            os.remove(new_tgt_path)
        os.rename(tgt_path, new_tgt_path)

print("重命名完成！") 