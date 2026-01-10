import argparse
import json
import os
import shutil

# ================= 配置区 =================
# 参数设置
N_TRAIN_VIEWS = 8  # 训练集取 8 张
N_TEST_VIEWS = 25  # 测试集取 25 张


# ==========================================

def get_sorted_frames(frames):
    """辅助函数：过滤掉非标准命名，并按数字排序"""
    valid_frames = []
    for frame in frames:
        # file_path 通常是 "./train/r_0" 或 "r_0"
        basename = os.path.basename(frame['file_path'])

        # 检查是否符合 r_{number} 格式
        if basename.startswith('r_') and basename.split('_')[-1].isdigit():
            valid_frames.append(frame)

    # 按数字排序
    valid_frames.sort(key=lambda x: int(os.path.basename(x['file_path']).split('_')[-1]))
    return valid_frames


def process_subset(source_root, target_root, source_json_path, target_json_path, target_img_dir, n_views, subset_name):
    """通用的处理函数 (Train/Test 共用)"""
    print(f"\n--- 正在处理 {subset_name} 集 ---")

    if not os.path.exists(source_json_path):
        print(f"警告: 找不到 {source_json_path}，跳过。")
        return

    with open(source_json_path, 'r') as f:
        data = json.load(f)

    # 1. 筛选并排序
    frames = get_sorted_frames(data['frames'])
    total = len(frames)

    # 2. 均匀采样
    if total >= n_views:
        step = total // n_views
        selected_frames = frames[::step][:n_views]
    else:
        selected_frames = frames  # 不够就全选

    print(f"[{subset_name}] 从 {total} 张中筛选出 {len(selected_frames)} 张 (r_数字 格式).")

    # 3. 生成新 JSON
    new_data = data.copy()
    new_data['frames'] = selected_frames

    with open(target_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    # 4. 复制图片
    os.makedirs(target_img_dir, exist_ok=True)
    count = 0
    for frame in selected_frames:
        # 处理路径：json里可能是 "./train/r_0"，我们要拼接成完整路径
        # 去掉 ./ 并补上后缀
        rel_path_no_ext = frame['file_path'].replace('./', '').replace('\\', '/')

        # 尝试匹配 png 或 jpg
        found = False
        for ext in ['.png', '.JPG', '.jpg']:
            src_path = os.path.join(source_root, rel_path_no_ext + ext)
            if os.path.exists(src_path):
                # 目标路径保持一致结构
                dst_path = os.path.join(target_root, rel_path_no_ext + ext)

                # 确保目标子文件夹存在 (比如 target/train/)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                shutil.copy(src_path, dst_path)
                found = True
                count += 1
                break

        if not found:
            print(f"错误: 找不到图片文件 {rel_path_no_ext}")

    print(f"[{subset_name}] 图片复制完成: {count} 张")


def process_scene(scene, source_base, target_base, n_train_views, n_test_views):
    source_root = os.path.join(source_base, scene)
    target_root = os.path.join(target_base, f"{scene}_8views")

    if not os.path.exists(source_root):
        print(f"跳过: 找不到场景目录 {source_root}")
        return

    if os.path.exists(target_root):
        print(f"清理旧目录: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root)

    process_subset(
        source_root=source_root,
        target_root=target_root,
        source_json_path=os.path.join(source_root, "transforms_train.json"),
        target_json_path=os.path.join(target_root, "transforms_train.json"),
        target_img_dir=os.path.join(target_root, "train"),
        n_views=n_train_views,
        subset_name="Train"
    )

    process_subset(
        source_root=source_root,
        target_root=target_root,
        source_json_path=os.path.join(source_root, "transforms_test.json"),
        target_json_path=os.path.join(target_root, "transforms_test.json"),
        target_img_dir=os.path.join(target_root, "test"),
        n_views=n_test_views,
        subset_name="Test"
    )

    print("\n✅ 数据集构建完成！")
    print(f"输出路径: {target_root}")


def main():
    parser = argparse.ArgumentParser(description="Build small Blender datasets for all scenes.")
    parser.add_argument("--source_base", required=True, type=str,
                        help="Blender dataset root (contains all scene folders).")
    parser.add_argument("--target_base", required=True, type=str,
                        help="Output root for processed scenes.")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Optional scene list; default uses standard 8 Blender scenes.")
    parser.add_argument("--train_views", type=int, default=N_TRAIN_VIEWS)
    parser.add_argument("--test_views", type=int, default=N_TEST_VIEWS)
    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

    for scene in scenes:
        process_scene(scene, args.source_base, args.target_base, args.train_views, args.test_views)


if __name__ == "__main__":
    main()
