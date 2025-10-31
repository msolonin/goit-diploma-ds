import pandas as pd
import re


def extract_feet_inches(text, label):
    match = re.search(fr'{label}.*?/\s*(\d+)\'\s*(\d+)"', text)
    if match:
        ft, inch = map(int, match.groups())
        return ft + inch / 12
    return None


def clean_key(key):
    """Convert text to lowercase_snake_case"""
    key = key.lower()
    key = re.sub(r'[^\w\s]', '', key)
    key = re.sub(r'\s+', '_', key)
    return key.strip('_')


def clean_value(val):
    """Convert strings like '1,600' → 1600, else keep as string"""
    val = val.strip()
    val = val.replace(',', '')
    if re.match(r'^\d+(\.\d+)?$', val):
        return float(val) if '.' in val else int(val)
    return val


def parse_main_entity_fixed(text):
    result = {}
    skip_headers = {
        'Tonnage', 'Cabins & Passenger Capacity', 'Engines & Performance',
        'Classification', 'Manufacturer Data'
    }
    needs_headers = {'Deadweight, t', 'Fuel capacity, l', 'Water tank, l', 'Max people', 'Cabins', 'Berths for guests',
                     'Bathrooms', 'Engine options', 'CE class', 'Cruising Speed, kt.', 'Max Speed, kt.',
                     'Hull type', 'Hull material', 'Design', 'Concept', 'Builder', 'Country', 'Series', 'Model', 'Period of manufacture',
                     'Displacement, t', 'Crew', 'Bathrooms', 'Model'}
    needs_headers_lower = [f.lower().strip() for f in needs_headers]
    lines = [l.strip() for l in text.splitlines() if l.strip() != '' and l.strip() not in skip_headers]
    i = 0
    while i < len(lines) - 1:
        if lines[i].lower().strip() not in needs_headers_lower:
            i += 1
            continue
        key = lines[i]
        val = lines[i + 1]
        result[clean_key(key)] = clean_value(val)
        i += 2
    return result


def get_feet_inches(text, label):
    match = re.search(fr'{label}.*?/ *(\d+)\'(?: *(\d+))?"?', text)
    if match:
        ft = int(match.group(1))
        inch = int(match.group(2)) if match.group(2) else 0
        return ft + inch / 12
    return None


# Function to extract all values
def extract_dimensions(text):
    return pd.Series({
        'loa_m': float(re.search(r'Length\s*([\d.]+)m', text).group(1)),
        'loa_ft': get_feet_inches(text, 'Length'),
        'beam_m': float(re.search(r'Beam\s*([\d.]+)m', text).group(1)),
        'beam_ft': get_feet_inches(text, 'Beam'),
        'draft_m': float(re.search(r'Draft\s*([\d.]+)m', text).group(1)),
        'draft_ft': get_feet_inches(text, 'Draft')
    })


if __name__ == '__main__':
    FILE_NAME = "data/boats_itboat_MY.csv"
    GOLD_FILE_NAME = "data/boats_itboat_MY_gold.csv"
    df = pd.read_csv(FILE_NAME)
    df['clean'] = df['main_chars'].str.replace(r'\s+', ' ', regex=True).str.replace('\xa0', ' ')
    df[['loa_m', 'loa_ft', 'beam_m', 'beam_ft', 'draft_m', 'draft_ft']] = df['clean'].apply(extract_dimensions)
    parsed = df['main_entity'].apply(parse_main_entity_fixed)
    all_keys = sorted({key for d in parsed for key in d.keys()})
    print(all_keys)
    df_parsed = pd.json_normalize(parsed)
    df_final = pd.concat([df, df_parsed], axis=1)

    df_final = df_final.drop(columns=['clean', 'main_entity', 'boat_options', 'main_chars'])
    df_final.to_csv(GOLD_FILE_NAME, index=False)
    print("Clean dataset saved as parsed_boat_data.csv")
