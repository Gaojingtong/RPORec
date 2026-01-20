import gzip
import json
import os
import sys
import fire
from tqdm import tqdm
import requests
import re
import html
import pandas as pd
from datasets import DatasetDict, Dataset


def _download(path: str, type: str = 'review', category: str = "CDs_and_Vinyl") -> str:
    """
    Downloads the raw data file from the specified URL and saves it locally.
    """
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    if type == 'review':
        url += f"benchmark/5core/rating_only/{category}.csv.gz"
    elif type == 'meta':
        url += f"raw/meta_categories/meta_{category}.jsonl.gz"
    else:
        raise ValueError(f"Invalid type: {type}")

    base_name = os.path.basename(url)
    local_path = os.path.join(path, base_name)
    if not os.path.exists(local_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as f, tqdm(
                    desc=f"Downloading {base_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    bar.update(len(chunk))
            print(f"Downloaded {base_name} successfully.")
        else:
            raise ValueError(f"Failed to download {base_name}. HTTP status code: {response.status_code}")
    else:
        print(f"{base_name} already exists. Skipping download.")
    return local_path


def _parse_gz(path: str, desc: str):
    with gzip.open(path, 'r') as g:
        for l in tqdm(g, unit='lines', desc=desc):
            yield json.loads(l.strip())


def clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def load_items(path: str):
    items = []
    items_ids = []
    max_len = 50
    invalid_num = 0

    for i, item in enumerate(_parse_gz(path, 'Loading items')):
        if 'title' not in item or item['title'] is None:
            invalid_num += 1
            continue
        if item['title'].find('<span id') > -1:
            invalid_num += 1
            continue
        if len(item['title'].split(" ")) > max_len:
            invalid_num += 1
            continue
        price = item.get('price', None)
        try:
            price = float(price)
        except:
            invalid_num += 1
            continue
        for feature, value in item.items():
            if isinstance(value, str):
                value = clean_text(value)
                item[feature] = value
        items.append(item)
        items_ids.append(item['parent_asin'])

    print(f"Loaded {len(items)} items, {i - len(items)} items removed")
    return items, items_ids


def item_filter(review_items: pd.DataFrame,
        K, start_time, end_time, min_loading_num=10000):
    print(f"from {start_time} to {end_time}")
    curr_review_items = review_items[(review_items['time'] >= start_time) & (review_items['time'] < end_time)].copy()


    while True:
        # filter with remove_users and remove_items
        users = curr_review_items.groupby('user_id').size().reset_index(name='count')
        items = curr_review_items.groupby('parent_asin').size().reset_index(name='count')
        users2remove = users[users['count'] < K]['user_id'].tolist()
        items2remove = items[items['count'] < K]['parent_asin'].tolist()
        curr_review_items = curr_review_items[
            ~((curr_review_items['user_id'].isin(users2remove)) | (curr_review_items['parent_asin'].isin(items2remove)))]

        num_reviews = len(curr_review_items)
        num_users = len(curr_review_items['user_id'].unique())
        num_items = len(curr_review_items['parent_asin'].unique())

        if len(users2remove) == 0 and len(items2remove) == 0:
            break

    if num_items < min_loading_num:
        start_datetime = pd.to_datetime(start_time)
        if start_datetime.year > review_items['time'].min().year:
            start_time = pd.to_datetime(start_time) - pd.DateOffset(months=3)
            start_time = start_time.strftime('%Y-%m-%d')

            print(
                "[After filtering] "
                f"users: {num_users}, "
                f"items: {num_items}, "
                f"reviews: {num_reviews}")
            print("Not enough items, try to get more items")
            return item_filter(review_items=review_items, K=K,
                               start_time=start_time, end_time=end_time, min_loading_num=min_loading_num)
        else:
            print('Not enough items, but already reached the minimum year')

    print('Data filtering done!')
    print(
        "[Final Stats] "
        f"users: {num_users}, "
        f"items: {num_items}, "
        f"reviews: {num_reviews}, "
        f"density: {num_reviews / (num_users * num_items)}")

    return curr_review_items


def gen_semantics(info,category):
    # CDs_and_Vinyl: title, average_rating, rating_number, description
    if category=="CDs_and_Vinyl":
        '''{'main_category': 'Digital Music', 'title': 'Release Some Tension', 'average_rating': 4.6, 'rating_number': 112,
         'features': [], 'description': ['Swv ~ Release Some Tension'], 'price': 12.05, 'store': 'SWV Format: Audio CD',
         'categories': ['CDs & Vinyl', 'Dance & Electronic', 'House'],
         'details': '{"Is Discontinued By Manufacturer": "No", "Product Dimensions": "5.62 x 4.92 x 0.33 inches; 3.84 Ounces", "Manufacturer": "Sony Legacy", "Item model number": "2013033", "Original Release Date": "1997", "Run time": "53 minutes", "Date First Available": "December 7, 2006", "Label": "Sony Legacy", "Number of discs": "1"}',
         'parent_asin': 'B000002X4C', 'subtitle': '', 'item_id': 1}'''
        info = info.to_dict()
        semantics = f"'title': {info.get('title', '')}\n 'average_rating': {info.get('average_rating', '')}\n 'rating_number': {info.get('rating_number', '')}\n 'description': {info.get('description', '')}"
        return semantics
    elif category == "Musical_Instruments":
        '''{'main_category': 'Musical Instruments',
         'title': '3 Mini Color Violin Fingering Tape for Fretboard Note Positions', 'average_rating': 4.6,
         'rating_number': 840, 'features': [], 'description': [
            'Finally, no more ugly masking tape!These plastic tapesmake it easier for students to learn fingering positions. This neat, durable tape can be easily and accurately placed on the fingerboard of any string instrument. Each package contains 1 roll of red, 1'],
         'price': 11.97, 'store': 'Long Beach Music',
         'categories': ['Musical Instruments', 'Instrument Accessories', 'General Accessories'],
         'details': '{"Item Weight": "0.5 Ounces", "Package Dimensions": "5 x 4 x 0.25 inches", "Item model number": "VC-8581-JC", "Best Sellers Rank": {"Musical Instruments": 2020, "General Musical Instrument Accessories": 111}, "Is Discontinued By Manufacturer": "No", "Date First Available": "May 10, 2011", "Number of Strings": "4", "Brand": "Long Beach Music", "Instrument": "Violin"}',
         'parent_asin': 'B00502CTPW', 'subtitle': '', 'item_id': 4}'''

        info = info.to_dict()
        semantics = f"'title': {info.get('title', '')}\n 'average_rating': {info.get('average_rating', '')}\n 'rating_number': {info.get('rating_number', '')}\n 'description': {info.get('description', '')}"
        return semantics
    elif category=="Video_Games":
        '''{'main_category': 'Video Games', 'title': 'NBA 2K17 - Early Tip Off Edition - PlayStation 4',
         'average_rating': 4.3, 'rating_number': 223,
         'features': ['The #1 rated NBA video game simulation series for the last 15 years (Metacritic).',
                      'The #1 selling NBA video game simulation series for the last 9 years (NPD).',
                      'Over 85 awards and nominations since the launch of PlayStation 4 & Xbox One.',
                      'BEST IN CLASS GAMEPLAY - 2K puts shot making in your hands like never before. Advanced Skill Shooting gives you complete control over the power and aim of your perimeter shots as well as your ability to finish inside the paint.',
                      'THE PRELUDE - Begin your MyCAREER on one of 10 licensed collegiate programs, available for free download one week prior to launch!',
                      'MyCAREER - It’s all-new and all about basketball in 2K17 – and you’re in control. Your on-court performance and career decisions lead to different outcomes as you determine your path through an immersive new narrative, featuring Michael B. Jordan. Additionally, new player controls give you unparalleled supremacy on the court.',
                      'USA BASKETBALL - Take the court as Team USA with Coach K on the sidelines, or relive the glory of the ’92 Dream Team. Earn USAB MyTEAM cards and gear up your MyPLAYER with official USAB wearables.',
                      'COLLEGE INTEGRATION - For the first time, play as college basketball legends with each school’s all-time greats team and MyTEAM cards.',
                      'LEAGUE EXPANSION - For the first time, customize your MyLEAGUE and MyGM experience with league expansion. Choose your expansion team names, logos and uniforms, and share them with the rest of the NBA 2K community. Your customized league comes complete with everything from Expansion Drafts to modified schedules and more to ensure an authentic NBA experience.',
                      '2K BEATS Imagine Dragons, Grimes, Noah “40” Shabib of OVO Sound and Michael B. Jordan curate another electric 2K soundtrack, featuring 50 songs.'],
         'description': [
             'Following the record-breaking launch of NBA 2K16, the NBA 2K franchise continues to stake its claim as the most authentic sports video game with NBA 2K17. As the franchise that “all sports video games should aspire to be” (GamesRadar), NBA 2K17 will take the game to new heights and continue to blur the lines between video game and reality.'],
         'price': 58.0, 'store': '2K', 'categories': ['Video Games', 'PlayStation 4', 'Games'],
         'details': '{"Release date": "September 16, 2016", "Best Sellers Rank": {"Video Games": 57637, "PlayStation 4 Games": 2886}, "Pricing": "The strikethrough price is the List Price. Savings represents a discount off the List Price.", "Product Dimensions": "0.4 x 5.3 x 6.6 inches; 1.6 Ounces", "Type of item": "Video Game", "Rated": "Everyone", "Item model number": "47793", "Is Discontinued By Manufacturer": "No", "Item Weight": "1.6 ounces", "Manufacturer": "2K Games", "Date First Available": "April 13, 2016"}',
         'parent_asin': 'B00Z9TLVK0', 'subtitle': '', 'item_id': 1}'''

        info = info.to_dict()
        semantics = f"'title': {info.get('title', '')}\n 'average_rating': {info.get('average_rating', '')}\n 'rating_number': {info.get('rating_number', '')}\n 'description': {info.get('description', '')}"
        return semantics
    else:
        info = info.to_dict()
        semantics = f"'title': {info.get('title', '')}\n 'average_rating': {info.get('average_rating', '')}\n 'rating_number': {info.get('rating_number', '')}\n 'description': {info.get('description', '')}"
        return semantics

def item_info_to_df(category, metadata, new_items, asin2id, add_pad_item=True):
    """
    ['main_category', 'title', 'subtitle', 'average_rating', 'rating_number',
       'features', 'description', 'price', 'store', 'categories', 'details',
       'parent_asin']
    """
    metadata_df = pd.DataFrame(metadata)
    metadata_df.drop([x for x in ["videos", "author", "bought_together", "images"] if x in metadata_df.columns], axis=1, inplace=True)
    # filter the items according to the new_items_asins
    metadata_df = metadata_df[metadata_df['parent_asin'].isin(new_items)]
    metadata_df.loc[:, 'item_id'] = metadata_df['parent_asin'].map(asin2id)

    metadata_df = metadata_df.astype({'rating_number': 'int'})
    metadata_df['details'] = metadata_df['details'].apply(json.dumps)

    metadata_df.loc[:, 'subtitle'] = metadata_df['subtitle'].apply(
        lambda x: '' if x is None or str(x).lower() == 'nan' or str(x).lower() == 'none' else str(x))

    metadata_df.loc[:, 'description'] = metadata_df['description'].apply(
        lambda x: list(set(x)) if isinstance(x, list) else [x] if isinstance(x, str) else [])

    metadata_df['features'] = metadata_df['features'].apply(lambda x: x if isinstance(x, list) else [])

    if add_pad_item:
        metadata_df.loc[len(metadata_df)] = {
            'main_category': 'pad_category',
            'title': 'pad_title',
            'subtitle': 'pad_subtitle',
            'average_rating': 0,
            'rating_number': 0,
            'features': [],
            'description': [],
            'price': 0,
            'store': 'pad_store',
            'categories': [],
            'details': json.dumps({}),
            'parent_asin': 'pad_asin',
            'item_id': 0
        }
    tqdm.pandas(desc='generating item semantics')
    metadata_df["item_msg"] = metadata_df.progress_apply(gen_semantics, axis=1, category=category)
    metadata_df = metadata_df[[col for col in metadata_df.columns if col in ['item_id', 'item_msg']]]
    # filter the items according to the new_items_asins
    metadata_df.sort_values(by='item_id', inplace=True)
    return metadata_df

def save_data(category, reviews, metadata,
              file_name, data_root_dir,
              window_size=10,
              index_item_with_pad=True,
              add_interaction_id=True
              ):
    new_reviews = reviews.to_dict(orient='records')
    items = reviews['parent_asin'].unique().tolist()
    asin2title = {item['parent_asin']: item['title'] for item in tqdm(
        metadata, desc="Creating asin2title mapping") if item['parent_asin'] in items}
    new_items = set()
    if index_item_with_pad:
        asin2title['pad_asin'] = 'pad_title'
        asin2id = {asin: idx + 1 for idx, asin in enumerate(asin2title.keys())}
        asin2id['pad_asin'] = 0
        new_items.add('pad_asin')
    else:
        asin2id = {item: idx for idx, item in enumerate(asin2title.keys())}

    interact = {}

    for review in new_reviews:
        user = review['user_id']
        item = review['parent_asin']
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
            }
        new_items.add(item)
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['rating'])
        interact[user]['timestamps'].append(review['timestamp'])

    interaction_list = []
    for key in interact.keys():
        items = interact[key]['items']
        ratings = interact[key]['ratings']
        timestamps = interact[key]['timestamps']

        all = list(zip(items, ratings, timestamps))
        res = sorted(all, key=lambda x: int(x[2]))
        items, ratings, timestamps = zip(*res)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)

        interact[key]['items'] = items
        interact[key]['ratings'] = ratings
        interact[key]['timestamps'] = timestamps
        interact[key]['item_ids'] = [asin2id[item] for item in items]
        interact[key]['title'] = [asin2title[item] for item in items]

        for i in range(1, len(items)):
            st = max(i - window_size, 0)
            assert i - st > 0, f"i: {i}, st: {st}"
            interaction_list.append(
                [key,
                 interact[key]['items'][st:i], interact[key]['items'][i],
                 interact[key]['item_ids'][st:i], interact[key]['item_ids'][i],
                 interact[key]['title'][st:i], interact[key]['title'][i],
                 interact[key]['ratings'][st:i], interact[key]['ratings'][i],
                 interact[key]['timestamps'][st:i], interact[key]['timestamps'][i],

                 ])
    print(f"interaction_list: {len(interaction_list)}")

    # split train val test
    interaction_list = sorted(interaction_list, key=lambda x: int(x[-1]))

    os.makedirs(data_root_dir, exist_ok=True)
    column_names = ['user_id',
                    'item_asins', 'item_asin',
                    'history_item_id', 'item_id',
                    'history_item_title', 'item_title',
                    'history_rating', 'rating',
                    'history_timestamp', 'timestamp',
                    ]

    if add_interaction_id:
        for i in range(len(interaction_list)):
            interaction_list[i].append(i)
        column_names.append('interaction_id')
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            pd.DataFrame(interaction_list[:int(len(interaction_list) * 0.8)],
                         columns=column_names)),
        'valid': Dataset.from_pandas(
            pd.DataFrame(interaction_list[int(len(interaction_list) * 0.8):int(len(interaction_list) * 0.9)],
                         columns=column_names)),
        'test': Dataset.from_pandas(
            pd.DataFrame(interaction_list[int(len(interaction_list) * 0.9):],
                         columns=column_names)),
        'item_info': Dataset.from_pandas(
            item_info_to_df(category, metadata, new_items, asin2id, add_pad_item=index_item_with_pad)),
    })
    dataset_dir = os.path.join(data_root_dir, file_name)
    dataset_dict.save_to_disk(dataset_dir)

    print(f"Train: {len(dataset_dict['train'])}, "
                f"Val: {len(dataset_dict['valid'])}, "
                f"Test: {len(dataset_dict['test'])}, "
                f"Items: {len(dataset_dict['item_info'])}")

def main(category: str = "CDs_and_Vinyl", K: int = 0,
         st_year: int = 2022, st_month: int = 10,
         ed_year: int = 2023, ed_month: int = 10,
         window_size: int = 20,
         output: bool = True,
         data_root_dir="../data",
         postfix=''):

    if not os.path.exists(data_root_dir):
        try:
            os.makedirs(data_root_dir)
        except OSError as e:
            print(f"{data_root_dir} cannot be created.")

    review_path = _download(data_root_dir, type='review', category=category)
    meta_path = _download(data_root_dir, type='meta', category=category)

    meta_items, meta_items_ids = load_items(meta_path)
    review_items = pd.read_csv(os.path.join(data_root_dir, f'{category}.csv.gz'), encoding='utf-8')
    # filter the reviews according to the items
    review_items = review_items[review_items['parent_asin'].isin(meta_items_ids)]
    review_items['time'] = pd.to_datetime(review_items['timestamp'], unit='ms')

    print(f"Dataset: {category} items: {len(meta_items)} reviews: {len(review_items)}")

    filtered_review_items = item_filter(review_items, K=K,
                                        start_time=f"{st_year}-{st_month}-01",
                                        end_time=f"{ed_year}-{ed_month}-01")

    file_name = f"{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}"
    if postfix:
        file_name += f"_{postfix}"

    if output:
        save_data(category, filtered_review_items, meta_items, file_name, data_root_dir, window_size=window_size)


if __name__ == '__main__':
    fire.Fire(main)
