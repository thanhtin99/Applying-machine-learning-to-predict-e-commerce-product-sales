import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def crawl_shopee_reviews(item_id, shop_id):
    url = f"https://shopee.com.my/api/v2/item/get_ratings"
    limit = 50
    offset = 0
    reviews = []

    # Tạo session 
    session = requests.Session()
    session.headers.update({"Cookie": "_gcl_au=xxxx; SPC_IA=-1; SPC_EC=-; SPC_F=xxxx; SPC_U=-; SPC_T_ID=xxxx; SPC_T_IV=xxxx; SPC_SI=xxxx; _ga=xxxx; _gid=xxxx; cto_lwid=xxxx; _fbp=xxxx; _hjid=xxxx; SPC_SIxxxx=xxxx"})

    # Lặp qua Các Trang Đánh Giá
    while True:
        params = {
            "itemid": item_id,
            "shopid": shop_id,
            "offset": offset,
            "limit": limit,
            "filter": "0",
            "flag": "0",
            "sort": "0",
            "append": "0",
            "before_bundle": "",
            "language": "en",
        }
        
        response = session.get(url, params=params).json()
        if response["error"]:
            print(f"Error: {response['error']}")
            break
        elif response["data"]["ratings"]:
            for rating in response["data"]["ratings"]:
                reviews.append(rating)
            offset += limit
        else:
            break

    return reviews

# Lưu Đánh Giá vào DataFrame
def save_to_csv(reviews, shop_id, item_id, csv_folder=''):
    csv_filename = f"{csv_folder}SR_{shop_id}_{item_id}.csv"
    df = pd.DataFrame(reviews)
    df.to_csv(csv_filename, index=False, encoding='utf-8')

# Hàm chạy đa luồng
def process_item(index, row):
    item_id = str(row['itemid'])
    shop_id = str(row['shopid'])
    session = requests.Session()
    reviews = crawl_shopee_reviews(item_id, shop_id)
    save_to_csv(reviews, shop_id, item_id, './data/raw/reviews/')
    print(f"Retrieved {len(reviews)} reviews for item {item_id} from shop {shop_id}.")

# Tạo hàm main để tùy chọn item muốn crawl
if __name__ == "__main__":
    # Đọc csv từ link /content/drive/MyDrive/Colab Notebooks/ttn.csv
    csv_path = r"./data/processed/merged_daily_data.csv"    
    # Chỉ chọn những cột cần thiết để giảm độ phức tạp
    selected_columns = ['itemid', 'shopid']  # Thêm tên các cột bạn quan tâm
    data = pd.read_csv(csv_path, usecols=selected_columns)
    data = data.drop_duplicates(subset=['itemid', 'shopid'])
    print(f"Total items: {len(data)}")

    # Sử dụng ThreadPoolExecutor để chạy đa luồng
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, index, row) for index, row in data.iterrows()]

        # Chờ tất cả các luồng hoàn thành
        for future in futures:
            future.result()
