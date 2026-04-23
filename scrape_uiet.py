import requests
from bs4 import BeautifulSoup
import os

urls = {
    "about":        "https://uiet.puchd.ac.in/?page_id=102",
    "courses":      "https://uiet.puchd.ac.in/?page_id=44",
    "admission":    "https://uiet.puchd.ac.in/?page_id=726",
    "fee":          "https://uiet.puchd.ac.in/?page_id=14671",
    "be_admission": "https://uiet.puchd.ac.in/?page_id=5555",
    "me_admission": "https://uiet.puchd.ac.in/?page_id=105",
    "phd_admission":"https://uiet.puchd.ac.in/?page_id=111",
    "mechanical":   "https://uiet.puchd.ac.in/?page_id=444",
    "departments":  "https://uiet.puchd.ac.in/?page_id=49",
    "pumeet":       "https://uiet.puchd.ac.in/?page_id=4042",
    "puleet":       "https://uiet.puchd.ac.in/?page_id=5943",
}

os.makedirs("knowledge", exist_ok=True)

for name, url in urls.items():
    print(f"Scraping: {name} ...")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # main content extract
        content = soup.find("div", class_="entry-content")

        if content:
            text = content.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)

            with open(f"knowledge/{name}.txt", "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  [OK] {name} saved ({len(text)} chars)")
        else:
            print(f"  [SKIP] {name} - no entry-content div found")

    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print("\nKnowledge scraping complete!")