import os
import shutil
import subprocess
import cv2
import numpy as np
import glob
from PIL import Image
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# === PATH SETUP ===
VITON_TEST_DIR = r"D:/computervision/virtual dressing room/VDR/inputset/test"
OPENPOSE_DIR = r"D:/computervision/virtual dressing room/VDR/openpose"
PGN_DIR = r"D:/computervision/virtual dressing room/VDR/CIHP_PGN"
PGN_CHECKPOINT = r"D:/computervision/virtual dressing room/VDR/checkpoints/CIHP_pgn"

sys.path.append(PGN_DIR)

# === STEPS ===
def make_dirs():
    for sub in ["cloth", "cloth-mask", "image", "image-parse", "openpose-img", "openpose-json"]:
        os.makedirs(os.path.join(VITON_TEST_DIR, sub), exist_ok=True)

def save_images(person_path, cloth_path):
    person_img = Image.open(person_path).convert("RGB")
    person_img.save(os.path.join(VITON_TEST_DIR, "image", "person.jpg"), "JPEG")

    cloth_img = Image.open(cloth_path).convert("RGB")
    cloth_save_path = os.path.join(VITON_TEST_DIR, "cloth", "cloth.jpg")
    cloth_img.save(cloth_save_path, "JPEG")

    return "person.jpg", "cloth.jpg"


def generate_cloth_mask(cloth_path, save_path):
    print("🧥 Generating cloth mask...")
    cloth_img = cv2.imread(cloth_path)
    gray = cv2.cvtColor(cv2.cvtColor(cloth_img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Save as JPG
    cv2.imwrite(save_path.replace(".png", ".jpg"), mask)


def run_openpose():
    print("🧍 Running OpenPose...")
    openpose_bin = os.path.join(OPENPOSE_DIR, "bin", "OpenPoseDemo.exe")
    cmd = [
        openpose_bin,
        "--image_dir", os.path.join(VITON_TEST_DIR, "image"),
        "--write_json", os.path.join(VITON_TEST_DIR, "openpose-json"),
        "--write_images", os.path.join(VITON_TEST_DIR, "openpose-img"),
        "--model_pose", "COCO",
        "--model_folder", os.path.join(OPENPOSE_DIR, "models"),
        "--display", "0",
        "--render_pose", "1",
        "--disable_blending"
    ]
    subprocess.run(cmd, check=True)

    # Rename keypoints JSON
    for file in os.listdir(os.path.join(VITON_TEST_DIR, "openpose-json")):
        if file.endswith(".json") and "person" in file:
            os.rename(
                os.path.join(VITON_TEST_DIR, "openpose-json", file),
                os.path.join(VITON_TEST_DIR, "openpose-json", "person_keypoints.json")
            )
            break

def generate_real_parse():
    print("🧠 Running PGN parsing (inf_pgn.py)...")
    
    subprocess.run([
        "python",
        os.path.join(PGN_DIR, "inf_pgn.py"),
        "-i", os.path.join(VITON_TEST_DIR, "image"),
        "-o", os.path.join(VITON_TEST_DIR, "image-parse")
    ], check=True)

    # Find cleaned mask from cihp_parsing_maps
    cleaned_mask_path = os.path.join(VITON_TEST_DIR, "image-parse", "cihp_parsing_maps", "person_mask_cleaned.png")

    if not os.path.exists(cleaned_mask_path):
        # Try to find any *_mask_cleaned.png if exact name not found
        for file in os.listdir(os.path.join(VITON_TEST_DIR, "image-parse", "cihp_parsing_maps")):
            if file.endswith("_mask_cleaned.png"):
                cleaned_mask_path = os.path.join(VITON_TEST_DIR, "image-parse", "cihp_parsing_maps", file)
                break

    # Final destination
    final_parse_path = os.path.join(VITON_TEST_DIR, "image-parse", "person.png")

    if os.path.exists(cleaned_mask_path):
        # Convert PNG to JPG and save as person.jpg
        mask_img = Image.open(cleaned_mask_path)
        mask_img.save(final_parse_path, "JPEG")
        print(f"✅ Saved parsed mask as {final_parse_path}")
    else:
        raise FileNotFoundError("❌ Cleaned parsing mask not found.")



    # Rename parse result if double dots or wrong name
    for file in glob.glob(os.path.join(VITON_TEST_DIR, "image-parse", "*.png")):
        if "person" in file and "person.png" not in file:
            os.rename(file, os.path.join(VITON_TEST_DIR, "image-parse", "person.png"))


def write_test_pair(person_name, cloth_name):
    print("✏️ Writing test_pairs.txt...")
    pair_path = os.path.join(VITON_TEST_DIR, "..", "test_pairs.txt")
    with open(pair_path, "w") as f:
        f.write(f"{person_name} {cloth_name}\n")

def main(person_img, cloth_img):
    print("🔧 Starting preprocessing...")
    make_dirs()

    person_name, cloth_name = save_images(person_img, cloth_img)

    generate_cloth_mask(
    os.path.join(VITON_TEST_DIR, "cloth", cloth_name),
    os.path.join(VITON_TEST_DIR, "cloth-mask", "cloth.jpg")
    )


    run_openpose()
    generate_real_parse()
    write_test_pair(person_name, cloth_name)
    print("✅ Preprocessing complete! Ready for try-on stage.")

if __name__ == "__main__":
    person_img = input("👤 Enter the full path to the person image: ").strip()
    cloth_img = input("👕 Enter the full path to the cloth image: ").strip()
    main(person_img, cloth_img)
