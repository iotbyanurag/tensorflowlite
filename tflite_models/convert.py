import re

def convert_pbtxt_to_labelmap(pbtxt_path, labelmap_path):
    with open(pbtxt_path, 'r') as f:
        content = f.read()

    # Use regex to find all label entries
    labels = re.findall(r'id: (\d+)\s+display_name: "(.*)"', content)

    # Sort by id to ensure they are in the correct order
    labels = sorted(labels, key=lambda x: int(x[0]))

    with open(labelmap_path, 'w') as f:
        for id_, name in labels:
            f.write(f"{name}\n")

convert_pbtxt_to_labelmap('mscoco_label_map.pbtxt', 'labelmap.txt')

