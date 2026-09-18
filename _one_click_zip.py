import os
from pathlib import Path
import zipfile


def zip_current_dir(
    zip_name: str = "output.zip",
    exclude_dirs: list[str] | None = None,
    exclude_files: list[str] | None = None,
) -> None:
    """
    将当前路径下的所有文件和文件夹压缩到指定的 zip 文件中。

    Args:
        zip_name:       输出的 zip 文件名，如 "result.zip"
        exclude_dirs:   需要排除的文件夹名称列表，如 ["__pycache__", ".git", "dataset"]
        exclude_files:  需要排除的文件名列表，如 ["output.zip", ".DS_Store"]
    """
    exclude_dirs = set(exclude_dirs or [])
    exclude_files = set(exclude_files or [])

    # 确保排除列表中的 zip 文件自身也被排除，避免自包含
    exclude_files.add(zip_name)

    current_dir = Path(".")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(current_dir):
            root_path = Path(root)

            # 原地修改 dirs，跳过被排除的文件夹（os.walk 会据此不再进入这些目录）
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue

                file_path = root_path / file
                # arcname 去掉开头的 "./"，使 zip 内部路径更干净
                arcname = str(file_path.relative_to(current_dir))
                zf.write(file_path, arcname)

    print(f"压缩完成 -> {zip_name}")



if __name__ == '__main__':
    # 创建一个新的 ZIP 文件
    with zipfile.ZipFile("result.zip", "w") as myzip:
        myzip.write("result.csv")

    zip_current_dir("submission_code.zip", ["dataset", ".vscode", ".venv"], ['result.zip', 'train_v3.csv', 'test_v3.csv'])
