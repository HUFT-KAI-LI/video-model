import argparse
import json
from pathlib import Path
import av
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, default=Path("data/raw/youku/raw_manifest.jsonl"))
p.add_argument("--output", type=Path, default=Path("outputs/data_review/contact_sheet.jpg"))
p.add_argument("--count", type=int, default=24)
a = p.parse_args()
rows = [json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
rows = rows[:a.count]
sheet = Image.new("RGB", (4 * 256, ((len(rows) + 3) // 4) * 170), "#202020")
draw = ImageDraw.Draw(sheet)
for i, row in enumerate(rows):
    with av.open(row["video"]) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        picture = frame.to_image()
        picture.thumbnail((256, 144))
        x, y = (i % 4) * 256, (i // 4) * 170
        sheet.paste(picture, (x, y))
        draw.text((x + 4, y + 146), f"{i:03d} | {row['duration']:.1f}s", fill="white")
a.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(a.output)
a.output.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(a.output.resolve())
