import os

def generate_imaginet_annotation(root_dir, output_file, synthetic_label=1,
                                 content_type=0, generator_group=1, specific_generator=0):
    """
    Generate ImagiNet-style annotation txt file for all .webp images under root_dir.

    Each line format:
      relative_path,synthetic_label,content_type,generator_group,specific_generator

    Args:
      root_dir: folder where your Sora images are stored (e.g. "Sora_data/")
      output_file: path to save the annotation txt (e.g. "annotations/sora_test.txt")
      synthetic_label: 1 for synthetic images (your case)
      content_type: int, e.g. 0 for photos
      generator_group: int, arbitrary but consistent for your data
      specific_generator: int, arbitrary but consistent for your data
    """
    entries = []
    # print(f"Generating annotations for {root_dir} into {output_file}")
    root_dir = os.path.abspath(root_dir)
    # print(f"Scanning {root_dir} for .webp images...")
    print(f"os.walk(root_dir): {os.walk(root_dir)}")
    for dirpath, _, filenames in os.walk(root_dir):
        print(f"inside generator")
        print(f"Looking in {dirpath} for .webp files")
        for f in filenames:
            if f.lower().endswith(".webp") or f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                print(f"currently in {dirpath} looking for {f}")
                full_path = os.path.join(dirpath, f)
                # Make relative path from the repo root (assumed current working dir)
                rel_path = os.path.relpath(full_path, os.getcwd())
                line = f"{rel_path},{synthetic_label},{content_type},{generator_group},{specific_generator}"
                entries.append(line)

    entries.sort()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as out_f:
        out_f.write("\n".join(entries) + "\n")
    print(f"Saved {len(entries)} entries to {output_file}")

if __name__ == "__main__":
    # Example usage - edit paths and labels as needed
    generate_imaginet_annotation(
        # root_dir="Sora_data",            # your folder of webp images
        # root_dir="sora-trainer/data/synthetic/",            # your folder of webp images
        root_dir="test_gen",            # your folder of webp images
        output_file="annotations/testgen_images.txt",
        synthetic_label=1,               # real images say 0=real, 1=synthetic
        content_type=1,                  # say 1=faces
        generator_group=1,               # arbitrary ID for Sora generator group
        specific_generator=0             # arbitrary ID for specific gen
    )

