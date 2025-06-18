from datasets import load_dataset, DownloadConfig
import random


# photos    = load_dataset("coco", split="train[:500]")     # 500 COCO

# Load 500 COCO 2017 “instances” (object detection) samples
# dl_cfg = DownloadConfig(timeout=600, max_retries=5)
print("Loading datasets...")
# photos = load_dataset(
#     "shunk031/MSCOCO",
#     year=2017,
#     coco_task="instances",
#     split="train[:500]"
# )


# faces     = load_dataset("nvlabs/ffhq-dataset", split="train[:500]")  # 500 FFHQ
paintings = load_dataset("huggan/wikiart", split="train[:500]")       # 500 WikiArt
# misc      = load_dataset("open_images_v7", split="train[:500]")       # 500 OpenImages

# Combine them into one list of (image_path, label):
real_entries = []
# for item in photos + faces + paintings + misc:
for item in  paintings:
    path = item["image"][0] if isinstance(item["image"], list) else item["image"]
    real_entries.append(f"{path},0,0,0,0")  # label=0

# Shuffle and write to annotations/real_calib.txt
random.shuffle(real_entries)
with open("annotations/real_calib.txt","w") as f:
    f.write("\n".join(real_entries))

