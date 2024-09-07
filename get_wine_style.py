from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import csv

client = OpenAI()
INPUT_FILE = "wine_titles.txt"
OUTPUT_FILE = "wine_title_style.csv"
TASK_TYPE = "name"  # could be variety or name
BATCH_SIZE = 100


class WineStyleOptions(str, Enum):
    FULL_BODIED_RED = "Full-Bodied Red Wines"
    MEDIUM_BODIED_RED = "Medium-Bodied Red Wines"
    LIGHT_BODIED_RED = "Light-Bodied Red Wines"
    ROSE = "Rosé Wines"
    FULL_BODIED_WHITE = "Full-Bodied White Wines"
    LIGHT_BODIED_WHITE = "Light-Bodied White Wines"
    AROMATIC_WHITE = "Aromatic White Wines"
    DESSERT_FORTIFIED = "Dessert & Fortified Wines"
    CHAMPAGNE_SPARKLING = "Champagne & Sparkling Wines"


class WineStyle(BaseModel):
    wine_name: str
    style: WineStyleOptions


class NameStyleList(BaseModel):
    name_style_list: list[WineStyle]


def get_wine_styles(wine_list):
    # works the same for variety
    prompt = "You are given an array of 100 wine names. Based on your knowledge about wines, determine style for each of them and write a list of pairs (name, style) keeping the name same as in input."

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": wine_list},
        ],
        response_format=NameStyleList,
    )

    return completion.choices[0].message.parsed


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    wines = file.read().splitlines()

for i in range(0, len(wines), BATCH_SIZE):
    batch = wines[i : i + BATCH_SIZE]
    batch_result = get_wine_styles(", ".join(batch))

    data = [
        {TASK_TYPE: wine.wine_name, "style": wine.style.value}
        for wine in batch_result.name_style_list
    ]

    df = pd.DataFrame(data)

    if i == 0:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w",
            header=True,
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
    else:
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="a",
            header=False,
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )

    print(f"Batch {i}-{i+BATCH_SIZE} done.")

print(f"Wine styles have been appended batch by batch to '{OUTPUT_FILE}'")
