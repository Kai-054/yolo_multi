import os
from pathlib import Path
import zipfile
import tarfile
import logging
import time
import yaml

# Constants
DATASETS_DIR = Path('datasets')
SETTINGS_YAML = 'settings.yaml'  # File cấu hình của bạn
LOGGER = logging.getLogger(__name__)

def check_dataset(dataset, autodownload=True):
    """
    Kiểm tra dataset từ file YAML, hỗ trợ tải xuống tự động nếu không tìm thấy dataset và xử lý format với `name_1` và `name_2`.

    Args:
        dataset (str): Đường dẫn đến dataset hoặc file YAML mô tả dataset.
        autodownload (bool, optional): Tự động tải dataset nếu không tìm thấy. Mặc định là True.

    Returns:
        dict: Thông tin dataset đã được phân tích và các đường dẫn liên quan.
    """
    # Kiểm tra file
    file = check_file(dataset)

    # Tải xuống và giải nén (nếu cần thiết)
    extract_dir = ""
    if zipfile.is_zipfile(file) or tarfile.is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # Đọc file YAML
    with open(file, 'r') as stream:
        data = yaml.safe_load(stream)  # dictionary

    # Kiểm tra các key bắt buộc
    for k in ["train", "val"]:
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # đổi key 'validation' thành 'val'
    
    if "name_1" not in data and "name_2" not in data:
        raise SyntaxError(f"{dataset} key missing ❌.\nEither 'name_1' (for classification) or 'name_2' (for detection) is required in all data YAMLs.")
    
    if "name_1" in data and "name_2" in data and len(data["name_1"]) != len(data["name_2"]):
        raise SyntaxError(f"{dataset} 'name_1' length {len(data['name_1'])} and 'name_2: {len(data['name_2'])}' must match.")

    # Thiết lập số lượng lớp học
    data["nc_1"] = len(data.get("name_1", []))
    data["nc_2"] = len(data.get("name_2", []))

    # Kiểm tra và xử lý tên các lớp
    data["name_1"] = check_class_names(data.get("name_1", []))
    data["name_2"] = check_class_names(data.get("name_2", []))

    # Thiết lập đường dẫn
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    data["path"] = path  # download scripts
    for k in ["train", "val", "test"]:
        if data.get(k):  # thêm đường dẫn
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Phân tích YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name without URL auth
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {DATASETS_DIR}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    
    check_font("Arial.ttf" if is_ascii(data["name_1"]) else "Arial.Unicode.ttf")  # download fonts if needed

    return data  # dictionary

# Ví dụ sử dụng
dataset_info = check_dataset("path/to/your/dataset.yaml")
print(dataset_info)
