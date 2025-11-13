#!/usr/bin/env python3
"""
脚本功能：将源文件夹中的所有CSV文件复制到目标文件夹
使用方法：python extract_csv_files.py <源文件夹路径> <目标文件夹路径>
"""

import os
import shutil
import argparse
from pathlib import Path


def extract_csv_files(source_dir, target_dir, recursive=True):
    """
    从源文件夹提取所有CSV文件到目标文件夹
    
    参数:
        source_dir: 源文件夹路径
        target_dir: 目标文件夹路径
        recursive: 是否递归搜索子文件夹（默认True）
    """
    # 转换为Path对象
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源文件夹是否存在
    if not source_path.exists():
        print(f"错误：源文件夹不存在: {source_dir}")
        return
    
    if not source_path.is_dir():
        print(f"错误：源路径不是文件夹: {source_dir}")
        return
    
    # 创建目标文件夹（如果不存在）
    if target_path.exists():
        # 清空目标文件夹
        print(f"清空目标文件夹: {target_path}")
        for item in target_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"警告：无法删除 {item}: {e}")
    else:
        target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"目标文件夹: {target_path}")
    
    # 查找所有CSV文件
    if recursive:
        csv_files = list(source_path.rglob("*.csv"))
    else:
        csv_files = list(source_path.glob("*.csv"))
    
    if not csv_files:
        print(f"在 {source_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("-" * 60)
    
    # 复制文件
    copied_count = 0
    for csv_file in csv_files:
        try:
            # 只复制文件名，不保留目录结构
            target_file = target_path / csv_file.name
            
            # 如果目标文件已存在，添加编号
            if target_file.exists():
                base_name = target_file.stem
                extension = target_file.suffix
                counter = 1
                while target_file.exists():
                    target_file = target_file.parent / f"{base_name}_{counter}{extension}"
                    counter += 1
            
            # 复制文件
            shutil.copy2(csv_file, target_file)
            print(f"✓ 已复制: {csv_file.name}")
            if csv_file.parent != source_path:
                print(f"  来源: {csv_file.relative_to(source_path)}")
            copied_count += 1
            
        except Exception as e:
            print(f"✗ 复制失败 {csv_file.name}: {e}")
    
    print("-" * 60)
    print(f"完成！成功复制 {copied_count}/{len(csv_files)} 个文件到 {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="将源文件夹中的所有CSV文件提取到目标文件夹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取所有CSV文件（包括子文件夹）
  python extract_csv_files.py /path/to/source /path/to/target
  
  # 只提取当前层级的CSV文件（不递归）
  python extract_csv_files.py /path/to/source /path/to/target --no-recursive
        """
    )
    
    parser.add_argument(
        "--source_dir",
        help="源文件夹路径（包含CSV文件的文件夹）"
    )
    
    parser.add_argument(
        "--target_dir",
        help="目标文件夹路径（CSV文件将被复制到这里）"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归搜索子文件夹，只提取顶层CSV文件"
    )
    
    args = parser.parse_args()
    
    # 执行提取
    extract_csv_files(
        args.source_dir,
        args.target_dir,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()

